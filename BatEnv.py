
import numpy as np
import pandas as pd
import pyomo.environ as en
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class BatEnv(MultiAgentEnv):
    """
    Multi-agent RL environment for a community with PV sharing and batteries.

    - Each step builds and solves a short Pyomo model (MPC style; default H=1).
    - Action per agent a in [0,1]: cap on fraction of its α-share of the residual PV pool
      that may go into battery at the CURRENT step. The remainder of its share is export.
    - Reward is negative energy cost at the current step.
    """

    def __init__(self, env_config):
        super().__init__()
        self.cfg = env_config
        self.com = env_config["community"]          # same object you pass to FlexEnv
        self.agents_id = list(self.com.agents.keys())
        self._agent_ids = set(self.agents_id)

        # timing
        pc = self.com.problem_conf
        self.step_min = int(pc["step_size"])        # minutes per step (e.g., 15)
        self.T = int(pc["t_end"] - pc["t_init"])    # dataset length (timesteps)
        self.Tw = int(pc.get("window_size", 96))    # episode window (fallback)
        self.tstep_init = int(pc["t_init"])
        self.tstep_end  = int(pc["t_end"])
        self.H = int(env_config.get("H", 1))        # MPC horizon (actions apply to t=0)
        self.price_sell = float(env_config.get("price_sell", 0.04))  # €/kWh (positive => cost term)

        # solver
        self.solver_name = env_config.get("solver", "glpk")

        # quick handles to raw series
        self._load = {aid: pd.Series(self.com.agents[aid].data["load"]) for aid in self.agents_id}
        self._pv   = {aid: pd.Series(self.com.agents[aid].data["gen"])  for aid in self.agents_id}
        self._tar  = {aid: pd.Series(self.com.agents[aid].tariff)       for aid in self.agents_id}

        # initial SOC state pulled from battery configs
        self.soc_state = {aid: self._initial_soc_kwh(aid) for aid in self.agents_id}

        # spaces
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)  # a-share cap
        # obs: [load, pv_raw, posLoad0, poolPV, soc, tariff]
        self.obs_dim = 6
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "observations": spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(self.obs_dim,), dtype=np.float32)
        })

        # episode bookkeeping
        self.tstep = self.tstep_init
        self.done_agents = {aid: False for aid in self.agents_id}
        self.R = {aid: 0.0 for aid in self.agents_id}

    # ------------------------------------------------------------------
    # RL API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        # reset time window
        self.tstep = self.tstep_init
        self.done_agents = {aid: False for aid in self.agents_id}
        self.R = {aid: 0.0 for aid in self.agents_id}
        # reset SOC from configs
        self.soc_state = {aid: self._initial_soc_kwh(aid) for aid in self.agents_id}
        return self._current_obs()

    def step(self, action_dict):
        """
        action_dict: {agent_id: np.array([a])} with a in [0,1]
        """
        # sanitize actions
        a_share = {aid: float(np.clip(np.array(action_dict.get(aid, [1.0]))[0], 0.0, 1.0))
                   for aid in self.agents_id}

        # build and solve model for window [tstep : tstep+H)
        model, results = self._solve_window(self.tstep, a_share)

        # per-agent immediate reward from Pyomo results at t=0
        rewards = {}
        for aid in self.agents_id:
            posNL = results["posNetLoad"][(aid, 0)]
            negNL = results["negNetLoad"][(aid, 0)]
            tar   = self._tar[aid].iloc[self.tstep]
            r = -(tar * posNL + self.price_sell * negNL)
            rewards[aid] = float(r)
            self.R[aid] += float(r)

            # advance SOC state with solved SOC at t=0
            self.soc_state[aid] = float(results["SOC"][(aid, 0)])

        # advance time
        self.tstep += 1

        # termination (episode ends at tstep_init + Tw or dataset end)
        done = (self.tstep >= min(self.tstep_init + self.Tw, self.tstep_end))
        self.done_agents = {aid: done for aid in self.agents_id}
        done_dict = {**self.done_agents, "__all__": done}

        return self._current_obs(), rewards, done_dict, {}

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _current_obs(self):
        obs = {}
        t = self.tstep
        # compute pool & posLoad0 for obs (self-consume first)
        posload0 = {}
        negload0 = {}
        for aid in self.agents_id:
            net = float(self._load[aid].iloc[t] - self._pv[aid].iloc[t])
            posload0[aid] = max(net, 0.0)
            negload0[aid] = max(-net, 0.0)
        poolPV_t = float(sum(negload0.values()))

        for aid in self.agents_id:
            o = np.array([
                float(self._load[aid].iloc[t]),
                float(self._pv[aid].iloc[t]),
                float(posload0[aid]),
                poolPV_t,
                float(self.soc_state[aid]),
                float(self._tar[aid].iloc[t]),
            ], dtype=np.float32)
            obs[aid] = {
                "action_mask": np.ones((1,), dtype=np.float32),
                "observations": o,
            }
        return obs

    # ------------------------------------------------------------------
    # Model build/solve (MPC window)
    # ------------------------------------------------------------------
    def _solve_window(self, t0, a_share):
        """
        Build and solve the Pyomo model over [t0 : t0+H). Uses current SOC as initial.
        Returns (model, results_dict_of_series_at_t).
        """
        H = min(self.H, self.tstep_end - t0)
        agent_ids = self.agents_id

        # --- assemble window arrays ---
        load_win = {(b, t): float(self._load[b].iloc[t0 + t]) for b in agent_ids for t in range(H)}
        pv_win   = {(b, t): float(self._pv[b].iloc[t0 + t])   for b in agent_ids for t in range(H)}
        tar_win  = {(b, t): float(self._tar[b].iloc[t0 + t])  for b in agent_ids for t in range(H)}

        # --- build model (mirrors your corrected battmodel_) ---
        m = en.ConcreteModel()
        m.BATTS = en.Set(initialize=agent_ids)
        m.Time  = en.RangeSet(0, H - 1)
        dt = self.step_min

        # Parameters
        m.priceBuy  = en.Param(m.BATTS, m.Time, initialize=tar_win)
        m.priceSell = en.Param(m.Time, initialize={t: self.price_sell for t in range(H)})

        # battery params pulled from com
        cap = {b: self.com.agents[b].conf["battery"]["capacity"] for b in agent_ids}
        pchg = {b: self.com.agents[b].conf["battery"]["charging_power_limit"] for b in agent_ids}
        pdis = {b: self.com.agents[b].conf["battery"]["discharging_power_limit"] for b in agent_ids}
        eta_c = {b: self.com.agents[b].conf["battery"]["charging_efficiency"] for b in agent_ids}
        eta_d = {b: self.com.agents[b].conf["battery"]["discharging_efficiency"] for b in agent_ids}

        m.Capacity         = en.Param(m.BATTS, initialize=cap)
        m.ChargingLimit    = en.Param(m.BATTS, initialize=pchg)
        m.DischargingLimit = en.Param(m.BATTS, initialize=pdis)
        m.etaChg           = en.Param(m.BATTS, initialize=eta_c)
        m.etaDisChg        = en.Param(m.BATTS, initialize=eta_d)

        # pre-sharing pos/neg load
        net = {(b, t): load_win[b, t] - pv_win[b, t] for b in agent_ids for t in range(H)}
        posload0 = {(b, t): max(net[b, t], 0.0) for b in agent_ids for t in range(H)}
        negload0 = {(b, t): max(-net[b, t], 0.0) for b in agent_ids for t in range(H)}
        m.posLoad0 = en.Param(m.BATTS, m.Time, initialize=posload0)
        m.negLoad0 = en.Param(m.BATTS, m.Time, initialize=negload0)
        m.PV       = en.Param(m.BATTS, m.Time, initialize=pv_win)  # for reference
        m.negLoad  = en.Param(m.BATTS, m.Time, initialize=negload0)  # back-compat

        pool = {t: sum(negload0[b, t] for b in agent_ids) for t in range(H)}
        m.poolPV = en.Param(m.Time, initialize=pool)

        # equal alpha
        alpha = {b: 1.0 / len(agent_ids) for b in agent_ids}
        m.alpha = en.Param(m.BATTS, initialize=alpha)

        # action-as-cap parameter (only t=0 uses the action; others default 1.0)
        a_cap = {(b, t): (a_share[b] if t == 0 else 1.0) for b in agent_ids for t in range(H)}
        m.aShare = en.Param(m.BATTS, m.Time, initialize=a_cap, within=en.PercentFraction)

        # decision variables
        m.SOC          = en.Var(m.BATTS, m.Time, bounds=lambda m, b, t: (0, m.Capacity[b]))
        m.posDeltaSOC  = en.Var(m.BATTS, m.Time,
                                bounds=lambda m, b, t: (0, m.ChargingLimit[b] * dt / 60.0))
        m.negDeltaSOC  = en.Var(m.BATTS, m.Time,
                                bounds=lambda m, b, t: (-abs(m.DischargingLimit[b]) * dt / 60.0, 0))
        m.posEInGrid   = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals)
        m.posEInPV     = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals)
        m.negEOutLocal = en.Var(m.BATTS, m.Time)  # usually ≤ 0 by balance
        m.negEOutExport= en.Var(m.BATTS, m.Time)  # forced = 0
        m.posNetLoad   = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals)
        m.negNetLoad   = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals)
        m.SharedToLoad = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals)

        # initial SOC set from env state
        soc0 = {b: self.soc_state[b] for b in agent_ids}
        def soc_rule(m, b, t):
            if t == 0:
                return m.SOC[b, t] == soc0[b] + m.posDeltaSOC[b, t] + m.negDeltaSOC[b, t]
            return m.SOC[b, t] == m.SOC[b, t-1] + m.posDeltaSOC[b, t] + m.negDeltaSOC[b, t]
        m.SOC_dyn = en.Constraint(m.BATTS, m.Time, rule=soc_rule)

        # exclusivity (big-M style via binaries optional; here we skip binaries for speed)

        # capacity feasibility
        def soc_discharge_limit(m, b, t):
            return m.negDeltaSOC[b, t] >= - (m.SOC[b, t-1] if t > 0 else soc0[b])
        m.SOC_dis_lim = en.Constraint(m.BATTS, m.Time, rule=soc_discharge_limit)

        def soc_charge_limit(m, b, t):
            return m.posDeltaSOC[b, t] <= m.Capacity[b] - (m.SOC[b, t-1] if t > 0 else soc0[b])
        m.SOC_ch_lim = en.Constraint(m.BATTS, m.Time, rule=soc_charge_limit)

        # battery energy balances
        def pos_in_rule(m, b, t):
            return m.posEInGrid[b, t] + m.posEInPV[b, t] == m.posDeltaSOC[b, t] / m.etaChg[b] * (60.0 / dt)
        m.pos_in = en.Constraint(m.BATTS, m.Time, rule=pos_in_rule)

        def neg_out_rule(m, b, t):
            return m.negEOutLocal[b, t] + m.negEOutExport[b, t] == m.negDeltaSOC[b, t] * m.etaDisChg[b] * (60.0 / dt)
        m.neg_out = en.Constraint(m.BATTS, m.Time, rule=neg_out_rule)

        # rate limits
        def chg_rate(m, b, t):
            return m.posEInGrid[b, t] + m.posEInPV[b, t] <= m.ChargingLimit[b]
        m.rate_chg = en.Constraint(m.BATTS, m.Time, rule=chg_rate)

        def dis_rate(m, b, t):
            return m.negEOutLocal[b, t] + m.negEOutExport[b, t] >= m.DischargingLimit[b]
        m.rate_dis = en.Constraint(m.BATTS, m.Time, rule=dis_rate)

        # sharing logic
        def shared_to_load_cap_rule(m, b, t):
            return m.SharedToLoad[b, t] <= m.posLoad0[b, t]
        m.shared_to_load_cap = en.Constraint(m.BATTS, m.Time, rule=shared_to_load_cap_rule)

        def pool_cap_rule(m, t):
            return sum(m.SharedToLoad[bb, t] for bb in m.BATTS) <= m.poolPV[t]
        m.pool_cap = en.Constraint(m.Time, rule=pool_cap_rule)

        m.posLoad = en.Expression(m.BATTS, m.Time,
                                  rule=lambda m, b, t: m.posLoad0[b, t] - m.SharedToLoad[b, t])

        # Residual split (exact) AND action-as-cap on PV→battery
        def residual_split_rule(m, b, t):
            residual = m.poolPV[t] - sum(m.SharedToLoad[bb, t] for bb in m.BATTS)
            return m.posEInPV[b, t] + m.negNetLoad[b, t] == m.alpha[b] * residual
        m.split = en.Constraint(m.BATTS, m.Time, rule=residual_split_rule)

        def action_cap_rule(m, b, t):
            residual = m.poolPV[t] - sum(m.SharedToLoad[bb, t] for bb in m.BATTS)
            return m.posEInPV[b, t] <= m.aShare[b, t] * m.alpha[b] * residual
        m.action_cap = en.Constraint(m.BATTS, m.Time, rule=action_cap_rule)

        # No export from battery
        def no_export_batt(m, b, t):
            return m.negEOutExport[b, t] == 0
        m.no_export = en.Constraint(m.BATTS, m.Time, rule=no_export_batt)

        # Local discharge cannot exceed remaining local load after sharing
        def local_dis_limit(m, b, t):
            return m.negEOutLocal[b, t] >= -m.posLoad[b, t]
        m.local_dis_lim = en.Constraint(m.BATTS, m.Time, rule=local_dis_limit)

        # Grid import accounting
        def pos_net_rule(m, b, t):
            return m.posNetLoad[b, t] == m.posLoad[b, t] + m.posEInGrid[b, t] + m.negEOutLocal[b, t]
        m.pos_net = en.Constraint(m.BATTS, m.Time, rule=pos_net_rule)

        # Objective: minimize energy cost
        def obj_rule(m):
            return sum(m.priceBuy[b, t] * m.posNetLoad[b, t] + m.priceSell[t] * m.negNetLoad[b, t]
                       for b in m.BATTS for t in m.Time)
        m.obj = en.Objective(rule=obj_rule, sense=en.minimize)

        # solve
        solver = en.SolverFactory(self.solver_name)
        _ = solver.solve(m, tee=False)

        # collect results at all t, but env uses t=0 for reward/updates
        results = {}
        for name in ["SOC", "posNetLoad", "negNetLoad", "SharedToLoad",
                     "posEInPV", "posEInGrid", "negEOutLocal"]:
            var = getattr(m, name)
            results[name] = {(b, t): float(var[b, t].value) for b in agent_ids for t in range(H)}

        return m, results

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    def _initial_soc_kwh(self, aid):
        """Pull initial SOC (kWh) from the agent's Battery config."""
        bconf = self.com.agents[aid].conf["battery"]
        cap = float(bconf["capacity"])
        # accept either absolute kWh or fraction in [0,1]
        if "current_charge" in bconf:
            return float(bconf["current_charge"])
        if "initial_charge" in bconf:
            return float(bconf["initial_charge"])
        if "initial_soc" in bconf:
            val = float(bconf["initial_soc"])
            return cap * val if 0.0 <= val <= 1.0 else float(val)
        # fallback
        return 0.5 * cap

import pyomo.environ as en
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from battery import Battery

def battmodel_(com, H):
    """
    com: Community instance
    H: Time horizon (number of time steps)
    """

    # α rule (equal split for now) — nested inside battmodel_
    def alpha(agent_ids):
        n = len(agent_ids)
        if n == 0:
            return {}
        w = 1.0 / n
        return {aid: w for aid in agent_ids}

    m = en.ConcreteModel()

    dt = 15  # minutes

    # Build battery objects
    battery_dict = {}
    for aid, agent in com.agents.items():
        batt_params = agent.conf['battery']
        battery = Battery(**batt_params)
        battery.set_initial_charge()
        battery_dict[aid] = battery

    # Data frames
    load_df = pd.DataFrame({aid: agent.data['load'] for aid, agent in com.agents.items()})
    pv_df   = pd.DataFrame({aid: agent.data['gen']  for aid, agent in com.agents.items()})

    # Prices
    buy_prices  = {(b, t): com.agents[b].tariff[t] for b in battery_dict for t in range(H)}
    sell_prices = {t: 0.04 for t in range(H)}  # keep sign convention as in your code

    # Sets
    agent_ids = list(battery_dict.keys())
    m.BATTS = en.Set(initialize=agent_ids)
    m.Time  = en.RangeSet(0, H - 1)

    # === Pre-sharing (self-consumption) shortage/surplus ===
    # net = load - pv; pos is remaining load after own PV, neg is own PV surplus
    net = {(b, t): float(load_df[b].iloc[t] - pv_df[b].iloc[t]) for b in com.agents for t in range(H)}
    posload0 = {(b, t): max(net[b, t], 0.0) for b in com.agents for t in range(H)}   # shortage after own PV
    negload0 = {(b, t): max(-net[b, t], 0.0) for b in com.agents for t in range(H)}  # own PV surplus

    m.posLoad0 = en.Param(m.BATTS, m.Time, initialize=posload0)
    m.negLoad0 = en.Param(m.BATTS, m.Time, initialize=negload0)
    # Back-compat for external scripts that expect 'negLoad'
    m.negLoad  = en.Param(m.BATTS, m.Time, initialize=negload0)

    # For reference (not strictly required)
    pv_dict = {(b, t): float(pv_df[b].iloc[t]) for b in com.agents for t in range(H)}
    m.PV = en.Param(m.BATTS, m.Time, initialize=pv_dict)

    # Community PV pool (sum of all individual surpluses)
    pool = {t: sum(negload0[b, t] for b in agent_ids) for t in range(H)}
    m.poolPV = en.Param(m.Time, initialize=pool)

    # Alpha weights (equal split for now)
    alpha_w = alpha(agent_ids)
    m.alpha = en.Param(m.BATTS, initialize=lambda m, b: alpha_w[b])

    # Prices
    m.priceSell = en.Param(m.Time, initialize=sell_prices)
    m.priceBuy  = en.Param(m.BATTS, m.Time, initialize=buy_prices)

    # Battery state and flows
    m.SOC = en.Var(m.BATTS, m.Time, bounds=lambda m, b, t: (0, battery_dict[b].capacity))
    m.posDeltaSOC = en.Var(
        m.BATTS, m.Time,
        bounds=lambda m, b, t: (0, battery_dict[b].charging_power_limit * dt / 60.0),
        initialize=0
    )
    m.negDeltaSOC = en.Var(
        m.BATTS, m.Time,
        bounds=lambda m, b, t: (-abs(battery_dict[b].discharging_power_limit) * dt / 60.0, 0),
        initialize=0
    )

    # Charging/discharging from/to where
    m.posEInGrid = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals, initialize=0)  # grid -> batt
    m.posEInPV   = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals, initialize=0)  # PV  -> batt

    m.negEOutLocal  = en.Var(m.BATTS, m.Time, initialize=0)   # batt -> local load  (<= 0 via balance)
    m.negEOutExport = en.Var(m.BATTS, m.Time, initialize=0)   # batt -> export      (forced 0 below)

    # Import/export accounting for objective
    m.posNetLoad = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals, initialize=0)  # grid import
    m.negNetLoad = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals, initialize=0)  # PV export after split

    # Binary charge/discharge exclusivity
    m.Bool_char = en.Var(m.BATTS, m.Time, within=en.Boolean)
    m.Bool_dis  = en.Var(m.BATTS, m.Time, within=en.Boolean, initialize=0)

    # Efficiencies and limits
    m.etaChg            = en.Param(m.BATTS, initialize=lambda m, b: battery_dict[b].charging_efficiency)
    m.etaDisChg         = en.Param(m.BATTS, initialize=lambda m, b: battery_dict[b].discharging_efficiency)
    m.ChargingLimit     = en.Param(m.BATTS, initialize=lambda m, b: battery_dict[b].charging_power_limit)
    m.DischargingLimit  = en.Param(m.BATTS, initialize=lambda m, b: battery_dict[b].discharging_power_limit)

    # === Community PV sharing variables ===
    m.SharedToLoad = en.Var(m.BATTS, m.Time, domain=en.NonNegativeReals, initialize=0)  # pool -> meet load
    # (Removed SharedToBattery slack variable)

    # SOC dynamics
    def soc_rule(m, b, t):
        if t == 0:
            return m.SOC[b, t] == battery_dict[b].current_charge + m.posDeltaSOC[b, t] + m.negDeltaSOC[b, t]
        return m.SOC[b, t] == m.SOC[b, t - 1] + m.posDeltaSOC[b, t] + m.negDeltaSOC[b, t]
    m.Batt_SOC = en.Constraint(m.BATTS, m.Time, rule=soc_rule)

    # Objective
    def Obj_fn(m):
        return sum(
            m.priceBuy[b, t] * m.posNetLoad[b, t] + m.priceSell[t] * m.negNetLoad[b, t]
            for b in m.BATTS for t in m.Time
        )
    m.total_cost = en.Objective(rule=Obj_fn, sense=en.minimize)

    # Mutually exclusive charge/discharge (big-M)
    def Bool_char_rule_1(m, b, t):
        return m.posDeltaSOC[b, t] >= -500000 * m.Bool_char[b, t]
    m.Batt_ch1 = en.Constraint(m.BATTS, m.Time, rule=Bool_char_rule_1)

    def Bool_char_rule_2(m, b, t):
        return m.posDeltaSOC[b, t] <= 500000 * (1 - m.Bool_dis[b, t])
    m.Batt_ch2 = en.Constraint(m.BATTS, m.Time, rule=Bool_char_rule_2)

    def Bool_char_rule_3(m, b, t):
        return m.negDeltaSOC[b, t] <= 500000 * m.Bool_dis[b, t]
    m.Batt_cd3 = en.Constraint(m.BATTS, m.Time, rule=Bool_char_rule_3)

    def Bool_char_rule_4(m, b, t):
        return m.negDeltaSOC[b, t] >= -500000 * (1 - m.Bool_char[b, t])
    m.Batt_cd4 = en.Constraint(m.BATTS, m.Time, rule=Bool_char_rule_4)

    # SOC feasibility vs capacity
    def soc_discharge_limit(m, b, t):
        if t == 0:
            return m.negDeltaSOC[b, t] >= -battery_dict[b].current_charge
        return m.negDeltaSOC[b, t] >= -m.SOC[b, t - 1]
    m.SOC_discharge_limit = en.Constraint(m.BATTS, m.Time, rule=soc_discharge_limit)

    def soc_charge_limit(m, b, t):
        if t == 0:
            return m.posDeltaSOC[b, t] <= battery_dict[b].capacity - battery_dict[b].current_charge
        return m.posDeltaSOC[b, t] <= m.capacity[b] - m.SOC[b, t - 1] if hasattr(m, 'capacity') else \
               m.posDeltaSOC[b, t] <= battery_dict[b].capacity - m.SOC[b, t - 1]
    m.SOC_charge_limit = en.Constraint(m.BATTS, m.Time, rule=soc_charge_limit)

    # No simultaneous charge & discharge
    def Batt_char_dis(m, b, t):
        return m.Bool_char[b, t] + m.Bool_dis[b, t] <= 1
    m.Batt_char_dis = en.Constraint(m.BATTS, m.Time, rule=Batt_char_dis)

    # Battery energy balance
    def pos_E_in_rule(m, b, t):
        return (m.posEInGrid[b, t] + m.posEInPV[b, t] == m.posDeltaSOC[b, t] / m.etaChg[b] * (60.0 / dt))
    m.posEIn_cons = en.Constraint(m.BATTS, m.Time, rule=pos_E_in_rule)

    def neg_E_out_rule(m, b, t):
        return (m.negEOutLocal[b, t] + m.negEOutExport[b, t] == m.negDeltaSOC[b, t] * m.etaDisChg[b] * (60.0 / dt))
    m.negEOut_cons = en.Constraint(m.BATTS, m.Time, rule=neg_E_out_rule)

    # Rate limits
    def E_charging_rate_rule(m, b, t):
        return (m.posEInGrid[b, t] + m.posEInPV[b, t]) <= m.ChargingLimit[b]
    m.chargingLimit_cons = en.Constraint(m.BATTS, m.Time, rule=E_charging_rate_rule)

    def E_discharging_rate_rule(m, b, t):
        return (m.negEOutLocal[b, t] + m.negEOutExport[b, t]) >= m.DischargingLimit[b]
    m.dischargingLimit_cons = en.Constraint(m.BATTS, m.Time, rule=E_discharging_rate_rule)

    # === Sharing logic ===

    # 1) You can only cover as much remaining load as your shortage after self-consumption
    def shared_to_load_cap_rule(m, b, t):
        return m.SharedToLoad[b, t] <= m.posLoad0[b, t]
    m.SharedToLoad_cap = en.Constraint(m.BATTS, m.Time, rule=shared_to_load_cap_rule)

    # 2) Total shared-to-load across agents cannot exceed the pool at each time
    def pool_cap_rule(m, t):
        return sum(m.SharedToLoad[bb, t] for bb in m.BATTS) <= m.poolPV[t]
    m.Pool_cap = en.Constraint(m.Time, rule=pool_cap_rule)

    # 3) Post-sharing load (shortage after applying shared pool)
    m.posLoad = en.Expression(m.BATTS, m.Time,
                              rule=lambda m, b, t: m.posLoad0[b, t] - m.SharedToLoad[b, t])

    # 4) Residual pool is split EXACTLY by alpha per agent into *actual* uses:
    #    PV to battery (posEInPV) + PV credited as export (negNetLoad)
    def residual_split_rule(m, b, t):
        return m.posEInPV[b, t] + m.negNetLoad[b, t] == m.alpha[b] * (
            m.poolPV[t] - sum(m.SharedToLoad[bb, t] for bb in m.BATTS)
        )
    m.ResidualSplit_cons = en.Constraint(m.BATTS, m.Time, rule=residual_split_rule)

    # Keep: No export directly from battery (as in your original)
    def E_No_export_Battery(m, b, t):
        return m.negEOutExport[b, t] == 0
    m.NoExportDischarging_cons = en.Constraint(m.BATTS, m.Time, rule=E_No_export_Battery)

    # Local discharge cannot exceed remaining local load after sharing
    def E_local_discharge_rule(m, b, t):
        return m.negEOutLocal[b, t] >= -m.posLoad[b, t]
    m.localDischargingLimit_cons = en.Constraint(m.BATTS, m.Time, rule=E_local_discharge_rule)

    # Grid import accounting uses post-sharing load
    def E_pos_net_rule(m, b, t):
        return m.posNetLoad[b, t] == m.posLoad[b, t] + m.posEInGrid[b, t] + m.negEOutLocal[b, t]
    m.E_posNet_cons = en.Constraint(m.BATTS, m.Time, rule=E_pos_net_rule)

    return m

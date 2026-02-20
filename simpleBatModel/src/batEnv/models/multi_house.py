from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Optional

import pyomo.environ as pyo


def _series_to_param_2d(houses: Sequence[str], series_by_house: Mapping[str, Sequence[float]]) -> dict[tuple[str, int], float]:
    """
    Build a {(house, t): value} dict for Pyomo Params over (H, T) with T indexed from 1..T.
    """
    any_house = next(iter(houses))
    T = len(series_by_house[any_house])
    out: dict[tuple[str, int], float] = {}
    for h in houses:
        s = series_by_house[h]
        if len(s) != T:
            raise ValueError("All houses must have the same horizon length")
        for k in range(T):
            out[(h, k + 1)] = float(s[k])
    return out


def _series_to_param_1d(arr: Sequence[float]) -> dict[int, float]:
    return {i + 1: float(arr[i]) for i in range(len(arr))}


@dataclass
class MultiHouseEnergySharingModel:
    """
    Multi-house MILP with:
      - Individual meters: each house has its own P_imp[h,t] and P_exp[h,t]
      - Individual batteries: each house has its own E[h,t], P_ch[h,t], P_dis[h,t]
      - Behind-the-scenes energy sharing between houses via a community bus:
          P_share[h,t] is a signed variable:
            >0 means house h sends power to the community
            <0 means house h receives power from the community
        Enforced by: sum_h P_share[h,t] == 0 for all t

      - PV curtailment per house (P_curt[h,t] >= 0)
      - Optional export prohibition (allow_export=False => P_exp == 0)

    Shared-energy price support (P2P settlement):
      - c_p2p_buy[t] : price paid by receivers (€/kWh)
      - c_p2p_sell[t]: price received by senders (€/kWh)
      - c_p2p_fee[t] : extra fee per traded kWh (€/kWh), applied once per trade (on P_share_out)

    Notes:
      - If settlement_in_objective=False (default), P2P buy/sell prices are *not* used in the optimization objective.
      - If settlement_in_objective=True, the spread (buy - sell) acts like a transaction cost and can change dispatch.
      - Fees c_p2p_fee always act as a transaction cost if non-zero.
    """

    dt: float
    allow_export: bool = True
    P_share_max: float = 0.0
    settlement_in_objective: bool = False

    def make_instance(
        self,
        *,
        loads_by_house: Mapping[str, Sequence[float]],
        pv_by_house: Mapping[str, Sequence[float]],
        battery_params_by_house: Mapping[str, Mapping[str, float]],
        c_grid: Sequence[float],
        c_sell: Sequence[float],
        c_p2p_buy: Optional[Sequence[float]] = None,
        c_p2p_sell: Optional[Sequence[float]] = None,
        c_p2p_fee: Optional[Sequence[float]] = None,
    ) -> pyo.ConcreteModel:
        houses = [str(h) for h in loads_by_house.keys()]
        houses.sort()
        if not houses:
            raise ValueError("loads_by_house is empty")

        T = len(next(iter(loads_by_house.values())))
        for h in houses:
            if len(loads_by_house[h]) != T:
                raise ValueError("All load series must have same length T")
            if len(pv_by_house[h]) != T:
                raise ValueError("All PV series must have same length T")
        if not (len(c_grid) == len(c_sell) == T):
            raise ValueError("Tariff series must have length T")

        # Default P2P series to zeros
        if c_p2p_buy is None:
            c_p2p_buy = [0.0] * T
        if c_p2p_sell is None:
            c_p2p_sell = [0.0] * T
        if c_p2p_fee is None:
            c_p2p_fee = [0.0] * T
        if not (len(c_p2p_buy) == len(c_p2p_sell) == len(c_p2p_fee) == T):
            raise ValueError("P2P price series must have length T (or be None)")

        # share max default: use max P_grid_max across houses if not specified
        if self.P_share_max and self.P_share_max > 0:
            share_max = float(self.P_share_max)
        else:
            share_max = max(float(battery_params_by_house[h]["P_grid_max"]) for h in houses)

        m = pyo.ConcreteModel()
        m.H = pyo.Set(initialize=houses, ordered=True)
        m.T = pyo.RangeSet(1, T)
        m.TE = pyo.RangeSet(0, T)

        m.dt = pyo.Param(initialize=float(self.dt))
        m.kappa_exp = pyo.Param(initialize=1 if bool(self.allow_export) else 0, within=pyo.Binary)

        # Tariffs
        m.c_grid = pyo.Param(m.T, initialize=_series_to_param_1d(c_grid), within=pyo.NonNegativeReals)
        m.c_sell = pyo.Param(m.T, initialize=_series_to_param_1d(c_sell), within=pyo.NonNegativeReals)

        # P2P prices (common)
        m.c_p2p_buy = pyo.Param(m.T, initialize=_series_to_param_1d(c_p2p_buy), within=pyo.NonNegativeReals)
        m.c_p2p_sell = pyo.Param(m.T, initialize=_series_to_param_1d(c_p2p_sell), within=pyo.NonNegativeReals)
        m.c_p2p_fee = pyo.Param(m.T, initialize=_series_to_param_1d(c_p2p_fee), within=pyo.NonNegativeReals)

        # Time series per house
        m.Load = pyo.Param(m.H, m.T, initialize=_series_to_param_2d(houses, loads_by_house), within=pyo.NonNegativeReals)
        m.PV = pyo.Param(m.H, m.T, initialize=_series_to_param_2d(houses, pv_by_house), within=pyo.NonNegativeReals)

        # Battery/grid params per house
        def _p(h, key):
            return float(battery_params_by_house[str(h)][key])

        m.E_init = pyo.Param(m.H, initialize={h: _p(h, "E_init") for h in houses})
        m.E_min = pyo.Param(m.H, initialize={h: _p(h, "E_min") for h in houses})
        m.E_max = pyo.Param(m.H, initialize={h: _p(h, "E_max") for h in houses})
        m.P_ch_max = pyo.Param(m.H, initialize={h: _p(h, "P_ch_max") for h in houses})
        m.P_dis_max = pyo.Param(m.H, initialize={h: _p(h, "P_dis_max") for h in houses})
        m.eta_ch = pyo.Param(m.H, initialize={h: _p(h, "eta_ch") for h in houses})
        m.eta_dis = pyo.Param(m.H, initialize={h: _p(h, "eta_dis") for h in houses})
        m.P_grid_max = pyo.Param(m.H, initialize={h: _p(h, "P_grid_max") for h in houses})

        # Vars (per house, per time)
        m.P_ch = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.P_dis = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.P_imp = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.P_exp = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.P_curt = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)

        # P2P sharing flows
        m.P_share_out = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals, bounds=(0.0, share_max))
        m.P_share_in = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals, bounds=(0.0, share_max))
        m.P_share = pyo.Var(m.H, m.T, within=pyo.Reals, bounds=(-share_max, share_max))

        m.E = pyo.Var(m.H, m.TE, within=pyo.NonNegativeReals)

        m.x = pyo.Var(m.H, m.T, within=pyo.Binary)
        m.y = pyo.Var(m.H, m.T, within=pyo.Binary)

        m.init_energy = pyo.Constraint(m.H, rule=lambda mm, h: mm.E[h, 0] == mm.E_init[h])

        def energy_dyn_rule(mm, h, t):
            return mm.E[h, t] == mm.E[h, t - 1] + mm.eta_ch[h] * mm.P_ch[h, t] * mm.dt - (1.0 / mm.eta_dis[h]) * mm.P_dis[h, t] * mm.dt

        m.energy_dyn = pyo.Constraint(m.H, m.T, rule=energy_dyn_rule)

        def energy_bounds_rule(mm, h, t):
            return pyo.inequality(mm.E_min[h], mm.E[h, t], mm.E_max[h])

        m.energy_bounds = pyo.Constraint(m.H, m.TE, rule=energy_bounds_rule)

        m.no_simul_batt_ch = pyo.Constraint(m.H, m.T, rule=lambda mm, h, t: mm.P_ch[h, t] <= mm.x[h, t] * mm.P_ch_max[h])
        m.no_simul_batt_dis = pyo.Constraint(m.H, m.T, rule=lambda mm, h, t: mm.P_dis[h, t] <= (1 - mm.x[h, t]) * mm.P_dis_max[h])

        m.no_simul_grid_imp = pyo.Constraint(m.H, m.T, rule=lambda mm, h, t: mm.P_imp[h, t] <= mm.y[h, t] * mm.P_grid_max[h])

        def no_simul_grid_exp_rule(mm, h, t):
            return mm.P_exp[h, t] <= (1 - mm.y[h, t]) * mm.P_grid_max[h] * mm.kappa_exp

        m.no_simul_grid_exp = pyo.Constraint(m.H, m.T, rule=no_simul_grid_exp_rule)

        m.curt_upper = pyo.Constraint(m.H, m.T, rule=lambda mm, h, t: mm.P_curt[h, t] <= mm.PV[h, t])

        # link signed share to in/out
        m.share_link = pyo.Constraint(m.H, m.T, rule=lambda mm, h, t: mm.P_share[h, t] == mm.P_share_out[h, t] - mm.P_share_in[h, t])

        def house_balance_rule(mm, h, t):
            return (
                mm.P_imp[h, t] + mm.PV[h, t] + mm.P_dis[h, t]
                == mm.Load[h, t] + mm.P_ch[h, t] + mm.P_exp[h, t] + mm.P_curt[h, t] + mm.P_share[h, t]
            )

        m.house_balance = pyo.Constraint(m.H, m.T, rule=house_balance_rule)

        m.community_balance = pyo.Constraint(m.T, rule=lambda mm, t: sum(mm.P_share[h, t] for h in mm.H) == 0.0)

        def obj_rule(mm):
            grid_cost = sum((mm.c_grid[t] * mm.P_imp[h, t] - mm.c_sell[t] * mm.P_exp[h, t]) * mm.dt for h in mm.H for t in mm.T)

            # fee counted ONCE per traded kWh (use outflows)
            fee_cost = sum(mm.c_p2p_fee[t] * mm.P_share_out[h, t] * mm.dt for h in mm.H for t in mm.T)

            if self.settlement_in_objective:
                settle_cost = sum(
                    (mm.c_p2p_buy[t] * mm.P_share_in[h, t] - mm.c_p2p_sell[t] * mm.P_share_out[h, t]) * mm.dt
                    for h in mm.H for t in mm.T
                )
                return grid_cost + fee_cost + settle_cost

            return grid_cost + fee_cost

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        return m

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pyomo.environ as pyo


def _to_2d_1_indexed_dict(
    by_house: Mapping[str, Sequence[float]],
    houses: Sequence[str],
    T: int,
) -> Dict[Tuple[str, int], float]:
    out: Dict[Tuple[str, int], float] = {}
    for h in houses:
        series = list(by_house[h])
        if len(series) != T:
            raise ValueError(f"Series for house '{h}' must have length T={T}")
        for t in range(1, T + 1):
            out[(h, t)] = float(series[t - 1])
    return out


@dataclass
class MultiHouseModel:
    """
    Unified house-indexed MILP with one independent battery per house.

    Two PV-allocation modes:
      alpha_mode="fixed"   — PV[h,t] is an exogenous Param (provided in
                              `pv_by_house`). This matches the legacy behaviour
                              and the four heuristic sharing strategies
                              (equal / weighted / consumption_instant / mean).
      alpha_mode="optimal" — PV[h,t] becomes a Var with the constraint
                              sum_h PV[h,t] == pv_total[t]. The MILP then
                              chooses the allocation that minimises cost. This
                              is the upper bound against which the heuristics
                              should be benchmarked.

    `cyclic_soc=True` (default) forces E[h, T] >= E_init[h] so the optimiser
    cannot 'cheat' by emptying the batteries on the last timestep — important
    for fair comparison across scenarios with finite horizons.

    Conventions:
      m.T  = RangeSet(1, T)         # decision timesteps
      m.TE = RangeSet(0, T)         # energy index (E[h,0] = E_init[h])
    """

    dt: float
    allow_export: bool = True
    cyclic_soc: bool = True

    def make_instance(
        self,
        *,
        houses: List[str],
        loads_by_house: Mapping[str, Sequence[float]],
        bat_params_by_house: Mapping[str, Mapping[str, Any]],
        c_grid: Sequence[float] | Mapping[str, Sequence[float]],
        c_sell: Sequence[float] | Mapping[str, Sequence[float]],
        pv_by_house: Optional[Mapping[str, Sequence[float]]] = None,
        pv_total: Optional[Sequence[float]] = None,
        alpha_mode: str = "fixed",
    ) -> pyo.ConcreteModel:
        if not houses:
            raise ValueError("houses must be a non-empty list")
        if alpha_mode not in ("fixed", "optimal"):
            raise ValueError(f"alpha_mode must be 'fixed' or 'optimal', got '{alpha_mode}'")

        houses = [str(h) for h in houses]
        T = len(next(iter(loads_by_house.values())))

        # ---------- input validation depending on alpha_mode ----------
        if alpha_mode == "fixed":
            if pv_by_house is None:
                raise ValueError("alpha_mode='fixed' requires pv_by_house")
            for h in houses:
                if h not in loads_by_house or h not in pv_by_house:
                    raise KeyError(f"Missing series for house '{h}'")
                if len(loads_by_house[h]) != T or len(pv_by_house[h]) != T:
                    raise ValueError(f"House '{h}' series must have length T={T}")
        else:  # alpha_mode == "optimal"
            if pv_total is None:
                raise ValueError("alpha_mode='optimal' requires pv_total")
            if len(pv_total) != T:
                raise ValueError(f"pv_total must have length T={T}")
            for h in houses:
                if h not in loads_by_house:
                    raise KeyError(f"Missing load series for house '{h}'")
                if len(loads_by_house[h]) != T:
                    raise ValueError(f"House '{h}' load series must have length T={T}")

        # ---------- tariff handling ----------
        if isinstance(c_grid, Mapping):
            c_grid_by_house = {h: list(c_grid[h]) for h in houses}
        else:
            c_grid_by_house = {h: list(c_grid) for h in houses}

        if isinstance(c_sell, Mapping):
            c_sell_by_house = {h: list(c_sell[h]) for h in houses}
        else:
            c_sell_by_house = {h: list(c_sell) for h in houses}

        for h in houses:
            if len(c_grid_by_house[h]) != T or len(c_sell_by_house[h]) != T:
                raise ValueError(f"Tariff series for house '{h}' must have length T={T}")

        # ---------- model construction ----------
        m = pyo.ConcreteModel()
        m.H = pyo.Set(initialize=houses, ordered=True)
        m.T = pyo.RangeSet(1, T)
        m.TE = pyo.RangeSet(0, T)

        m.dt = pyo.Param(initialize=float(self.dt))
        m.kappa_exp = pyo.Param(initialize=1 if bool(self.allow_export) else 0, within=pyo.Binary)

        def _p(h: str, k: str, default: float) -> float:
            return float((bat_params_by_house.get(h) or {}).get(k, default))

        m.E_init = pyo.Param(m.H, initialize=lambda mm, h: _p(h, "E_init", 0.0))
        m.E_min = pyo.Param(m.H, initialize=lambda mm, h: _p(h, "E_min", 0.0))
        m.E_max = pyo.Param(m.H, initialize=lambda mm, h: _p(h, "E_max", 0.0))
        m.P_ch_max = pyo.Param(m.H, initialize=lambda mm, h: _p(h, "P_ch_max", 0.0))
        m.P_dis_max = pyo.Param(m.H, initialize=lambda mm, h: _p(h, "P_dis_max", 0.0))
        m.eta_ch = pyo.Param(m.H, initialize=lambda mm, h: _p(h, "eta_ch", 1.0))
        m.eta_dis = pyo.Param(m.H, initialize=lambda mm, h: _p(h, "eta_dis", 1.0))
        m.P_grid_max = pyo.Param(m.H, initialize=lambda mm, h: _p(h, "P_grid_max", 1e9))

        m.Load = pyo.Param(m.H, m.T,
                           initialize=_to_2d_1_indexed_dict(loads_by_house, houses, T),
                           within=pyo.NonNegativeReals)
        m.c_grid = pyo.Param(m.H, m.T,
                             initialize=_to_2d_1_indexed_dict(c_grid_by_house, houses, T),
                             within=pyo.NonNegativeReals)
        m.c_sell = pyo.Param(m.H, m.T,
                             initialize=_to_2d_1_indexed_dict(c_sell_by_house, houses, T),
                             within=pyo.NonNegativeReals)

        # ---------- PV: Param vs Var depending on alpha_mode ----------
        m.alpha_mode = pyo.Param(initialize=alpha_mode, within=pyo.Any)
        if alpha_mode == "fixed":
            m.PV = pyo.Param(m.H, m.T,
                             initialize=_to_2d_1_indexed_dict(pv_by_house, houses, T),
                             within=pyo.NonNegativeReals)
        else:
            # PV becomes a decision variable bounded by the per-timestep total.
            pv_total_dict = {t + 1: float(pv_total[t]) for t in range(T)}
            m.PV_total = pyo.Param(m.T, initialize=pv_total_dict, within=pyo.NonNegativeReals)
            m.PV = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
            m.alpha_total = pyo.Constraint(
                m.T,
                rule=lambda mm, t: sum(mm.PV[h, t] for h in mm.H) == mm.PV_total[t],
            )
            m.alpha_bound = pyo.Constraint(
                m.H, m.T,
                rule=lambda mm, h, t: mm.PV[h, t] <= mm.PV_total[t],
            )

        # ---------- decision variables ----------
        m.P_ch = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.P_dis = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.P_imp = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.P_exp = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.P_curt = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
        m.E = pyo.Var(m.H, m.TE, within=pyo.NonNegativeReals)

        m.x = pyo.Var(m.H, m.T, within=pyo.Binary)  # 1 = charging
        m.y = pyo.Var(m.H, m.T, within=pyo.Binary)  # 1 = importing

        # ---------- constraints ----------
        def energy_dyn_rule(mm, h, t):
            return mm.E[h, t] == mm.E[h, t - 1] + (
                mm.eta_ch[h] * mm.P_ch[h, t] - (1.0 / mm.eta_dis[h]) * mm.P_dis[h, t]
            ) * mm.dt

        m.energy_dyn = pyo.Constraint(m.H, m.T, rule=energy_dyn_rule)
        m.energy_init = pyo.Constraint(m.H, rule=lambda mm, h: mm.E[h, 0] == mm.E_init[h])
        m.energy_bounds = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: pyo.inequality(mm.E_min[h], mm.E[h, t], mm.E_max[h]),
        )

        # Cyclic SoC: prevent the optimiser from emptying batteries at horizon end.
        if self.cyclic_soc:
            m.energy_final = pyo.Constraint(
                m.H, rule=lambda mm, h: mm.E[h, T] >= mm.E_init[h],
            )

        m.no_simul_ch = pyo.Constraint(
            m.H, m.T, rule=lambda mm, h, t: mm.P_ch[h, t] <= mm.P_ch_max[h] * mm.x[h, t],
        )
        m.no_simul_dis = pyo.Constraint(
            m.H, m.T, rule=lambda mm, h, t: mm.P_dis[h, t] <= mm.P_dis_max[h] * (1 - mm.x[h, t]),
        )

        m.no_simul_imp = pyo.Constraint(
            m.H, m.T, rule=lambda mm, h, t: mm.P_imp[h, t] <= mm.P_grid_max[h] * mm.y[h, t],
        )
        m.no_simul_exp = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: mm.P_exp[h, t] <= mm.P_grid_max[h] * mm.kappa_exp * (1 - mm.y[h, t]),
        )

        m.curt_limit = pyo.Constraint(
            m.H, m.T, rule=lambda mm, h, t: mm.P_curt[h, t] <= mm.PV[h, t],
        )

        def power_balance_rule(mm, h, t):
            return mm.Load[h, t] == (
                mm.PV[h, t] - mm.P_curt[h, t]
                + mm.P_dis[h, t] - mm.P_ch[h, t]
                + mm.P_imp[h, t] - mm.P_exp[h, t]
            )

        m.power_balance = pyo.Constraint(m.H, m.T, rule=power_balance_rule)

        m.obj = pyo.Objective(
            expr=sum(
                (m.c_grid[h, t] * m.P_imp[h, t] - m.c_sell[h, t] * m.P_exp[h, t]) * m.dt
                for h in m.H for t in m.T
            ),
            sense=pyo.minimize,
        )
        return m


# Backward-compatible alias kept for legacy imports.
MultiHouseEnergySharingModel = MultiHouseModel

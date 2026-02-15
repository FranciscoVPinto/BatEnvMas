from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping, Any

import pyomo.environ as pyo


def _to_1_indexed_dict(arr: Sequence[float]) -> dict[int, float]:
    # Pyomo RangeSet(1, T) friendly indexing
    return {i + 1: float(arr[i]) for i in range(len(arr))}


@dataclass
class SimpleBatteryModel:
    """
    Base MILP for a single house, equivalent to the current simple model.
    This class only builds a Pyomo ConcreteModel from already-loaded series.
    """

    dt: float  # hours

    E_init: float
    E_min: float
    E_max: float
    P_ch_max: float
    P_dis_max: float
    eta_ch: float
    eta_dis: float

    P_grid_max: float

    def make_instance(
        self,
        load: Sequence[float],
        pv: Sequence[float],
        c_grid: Sequence[float],
        c_sell: Sequence[float],
    ) -> pyo.ConcreteModel:
        T = len(load)
        if not (len(pv) == len(c_grid) == len(c_sell) == T):
            raise ValueError("Series length mismatch: load, pv, c_grid, c_sell must have same length")

        m = pyo.ConcreteModel()

        m.T = pyo.RangeSet(1, T)
        m.TE = pyo.RangeSet(0, T)

        m.dt = pyo.Param(initialize=float(self.dt))
        m.E_init = pyo.Param(initialize=float(self.E_init))
        m.E_min = pyo.Param(initialize=float(self.E_min))
        m.E_max = pyo.Param(initialize=float(self.E_max))
        m.P_ch_max = pyo.Param(initialize=float(self.P_ch_max))
        m.P_dis_max = pyo.Param(initialize=float(self.P_dis_max))
        m.eta_ch = pyo.Param(initialize=float(self.eta_ch))
        m.eta_dis = pyo.Param(initialize=float(self.eta_dis))
        m.P_grid_max = pyo.Param(initialize=float(self.P_grid_max))

        m.Load = pyo.Param(m.T, initialize=_to_1_indexed_dict(load), within=pyo.NonNegativeReals)
        m.PV = pyo.Param(m.T, initialize=_to_1_indexed_dict(pv), within=pyo.NonNegativeReals)
        m.c_grid = pyo.Param(m.T, initialize=_to_1_indexed_dict(c_grid), within=pyo.NonNegativeReals)
        m.c_sell = pyo.Param(m.T, initialize=_to_1_indexed_dict(c_sell), within=pyo.NonNegativeReals)

        m.P_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_imp = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_exp = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.E = pyo.Var(m.TE, within=pyo.NonNegativeReals)

        m.x = pyo.Var(m.T, within=pyo.Binary)  # charge vs discharge
        m.y = pyo.Var(m.T, within=pyo.Binary)  # import vs export

        def init_energy_rule(mm):
            return mm.E[0] == mm.E_init

        m.init_energy = pyo.Constraint(rule=init_energy_rule)

        def energy_dyn_rule(mm, t):
            return mm.E[t] == mm.E[t - 1] + mm.eta_ch * mm.P_ch[t] * mm.dt - (1.0 / mm.eta_dis) * mm.P_dis[t] * mm.dt

        m.energy_dyn = pyo.Constraint(m.T, rule=energy_dyn_rule)

        def energy_bounds_rule(mm, t):
            return pyo.inequality(mm.E_min, mm.E[t], mm.E_max)

        m.energy_bounds = pyo.Constraint(m.TE, rule=energy_bounds_rule)

        def no_simul_batt_ch_rule(mm, t):
            return mm.P_ch[t] <= mm.x[t] * mm.P_ch_max

        def no_simul_batt_dis_rule(mm, t):
            return mm.P_dis[t] <= (1 - mm.x[t]) * mm.P_dis_max

        m.no_simul_batt_ch = pyo.Constraint(m.T, rule=no_simul_batt_ch_rule)
        m.no_simul_batt_dis = pyo.Constraint(m.T, rule=no_simul_batt_dis_rule)

        def no_simul_grid_imp_rule(mm, t):
            return mm.P_imp[t] <= mm.y[t] * mm.P_grid_max

        def no_simul_grid_exp_rule(mm, t):
            return mm.P_exp[t] <= (1 - mm.y[t]) * mm.P_grid_max

        m.no_simul_grid_imp = pyo.Constraint(m.T, rule=no_simul_grid_imp_rule)
        m.no_simul_grid_exp = pyo.Constraint(m.T, rule=no_simul_grid_exp_rule)

        def power_balance_rule(mm, t):
            return mm.P_imp[t] + mm.PV[t] + mm.P_dis[t] == mm.Load[t] + mm.P_ch[t] + mm.P_exp[t]

        m.power_balance = pyo.Constraint(m.T, rule=power_balance_rule)

        def obj_rule(mm):
            return sum((mm.c_grid[t] * mm.P_imp[t] - mm.c_sell[t] * mm.P_exp[t]) * mm.dt for t in mm.T)

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        return m

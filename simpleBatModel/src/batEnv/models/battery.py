from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pyomo.environ as pyo


def _to_1_indexed_dict(arr: Sequence[float]) -> dict[int, float]:
    # Pyomo RangeSet(1, T) friendly indexing
    return {i + 1: float(arr[i]) for i in range(len(arr))}


@dataclass
class SimpleBatteryModel:
    """
    Single-house MILP (Load + PV + Battery + Grid), with:
      - no simultaneous charge/discharge (binary x[t])
      - no simultaneous import/export (binary y[t])
      - PV curtailment (P_curt[t] >= 0)
      - optional export prohibition via allow_export (forces P_exp[t] == 0)

    Notes:
      - All time-series inputs must have length T.
      - Power variables are in kW, energy in kWh, dt in hours.
    """

    dt: float  # hours

    # Battery parameters (kWh, kW)
    E_init: float
    E_min: float
    E_max: float
    P_ch_max: float
    P_dis_max: float
    eta_ch: float
    eta_dis: float

    # Grid big-M (kW) used for import/export limits
    P_grid_max: float

    # If False, export is prohibited (P_exp[t] == 0)
    allow_export: bool = True

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

        # Export enable switch (0/1). When 0, export is forced to 0.
        m.kappa_exp = pyo.Param(initialize=1 if bool(self.allow_export) else 0, within=pyo.Binary)

        m.Load = pyo.Param(m.T, initialize=_to_1_indexed_dict(load), within=pyo.NonNegativeReals)
        m.PV = pyo.Param(m.T, initialize=_to_1_indexed_dict(pv), within=pyo.NonNegativeReals)
        m.c_grid = pyo.Param(m.T, initialize=_to_1_indexed_dict(c_grid), within=pyo.NonNegativeReals)
        m.c_sell = pyo.Param(m.T, initialize=_to_1_indexed_dict(c_sell), within=pyo.NonNegativeReals)

        # Decision variables (kW)
        m.P_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_imp = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_exp = pyo.Var(m.T, within=pyo.NonNegativeReals)

        # PV curtailment (kW): PV available but not used (wasted)
        m.P_curt = pyo.Var(m.T, within=pyo.NonNegativeReals)

        # State of charge (kWh)
        m.E = pyo.Var(m.TE, within=pyo.NonNegativeReals)

        # Binary switches
        m.x = pyo.Var(m.T, within=pyo.Binary)  # charge vs discharge
        m.y = pyo.Var(m.T, within=pyo.Binary)  # import vs export

        # Initial energy
        m.init_energy = pyo.Constraint(expr=m.E[0] == m.E_init)

        # Battery dynamics
        def energy_dyn_rule(mm, t):
            return mm.E[t] == mm.E[t - 1] + mm.eta_ch * mm.P_ch[t] * mm.dt - (1.0 / mm.eta_dis) * mm.P_dis[t] * mm.dt

        m.energy_dyn = pyo.Constraint(m.T, rule=energy_dyn_rule)

        # Energy bounds (including t=0)
        def energy_bounds_rule(mm, t):
            return pyo.inequality(mm.E_min, mm.E[t], mm.E_max)

        m.energy_bounds = pyo.Constraint(m.TE, rule=energy_bounds_rule)

        # Prevent simultaneous charge and discharge
        m.no_simul_batt_ch = pyo.Constraint(m.T, rule=lambda mm, t: mm.P_ch[t] <= mm.x[t] * mm.P_ch_max)
        m.no_simul_batt_dis = pyo.Constraint(m.T, rule=lambda mm, t: mm.P_dis[t] <= (1 - mm.x[t]) * mm.P_dis_max)

        # Prevent simultaneous import and export
        m.no_simul_grid_imp = pyo.Constraint(m.T, rule=lambda mm, t: mm.P_imp[t] <= mm.y[t] * mm.P_grid_max)

        def no_simul_grid_exp_rule(mm, t):
            # if kappa_exp == 0 => P_exp[t] <= 0 => P_exp[t] == 0 (since NonNegative)
            return mm.P_exp[t] <= (1 - mm.y[t]) * mm.P_grid_max * mm.kappa_exp

        m.no_simul_grid_exp = pyo.Constraint(m.T, rule=no_simul_grid_exp_rule)

        # Curtailment cannot exceed available PV (tightens model)
        m.curt_upper = pyo.Constraint(m.T, rule=lambda mm, t: mm.P_curt[t] <= mm.PV[t])

        # Power balance (kW): inflows = outflows
        def power_balance_rule(mm, t):
            return mm.P_imp[t] + mm.PV[t] + mm.P_dis[t] == mm.Load[t] + mm.P_ch[t] + mm.P_exp[t] + mm.P_curt[t]

        m.power_balance = pyo.Constraint(m.T, rule=power_balance_rule)

        # Objective: minimize net cost
        m.obj = pyo.Objective(expr=sum((m.c_grid[t] * m.P_imp[t] - m.c_sell[t] * m.P_exp[t]) * m.dt for t in m.T), sense=pyo.minimize)

        return m

"""
MultiHouseModelDegradation — extension of MultiHouseModel that adds a linear
battery-degradation penalty to the objective.

This module deliberately leaves `MultiHouseModel` untouched. It builds the base
MILP via composition, then rewrites the Objective to include a throughput
penalty:

    Objective = original_grid_cost + sum_h sum_t lambda_deg[h] * (P_ch[h,t] + P_dis[h,t]) * dt

`lambda_deg` is the amortised cost per kWh that flows through the battery
(charge or discharge), expressed in EUR/kWh. A typical derivation is:

    lambda_deg = C_battery / (N_cycles * 2 * E_max)

  e.g. C_battery = 4000 EUR for a 10 kWh pack with 6000 cycles
       => lambda_deg = 4000 / (6000 * 2 * 10) = ~0.033 EUR/kWh

This term discourages 'free' cycling: the optimiser only charges/discharges
when the operational saving exceeds the wear cost. Result: the comparison
between sharing strategies becomes more realistic when batteries are not
considered free assets.

Both lambda is applied symmetrically to charge and discharge — reasonable for
domestic LFP/Li-ion. For more elaborate degradation models (rainflow counting,
SoC-dependent fade, calendar ageing) extend this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import pyomo.environ as pyo

from .multi_house import MultiHouseModel


@dataclass
class MultiHouseModelDegradation:
    """
    Drop-in replacement for `MultiHouseModel.make_instance` that adds a linear
    battery-degradation penalty to the cost objective.

    Parameters
    ----------
    dt, allow_export, cyclic_soc
        Forwarded to the underlying MultiHouseModel.
    lambda_deg
        Default per-kWh degradation cost (EUR/kWh). Applied to BOTH P_ch and
        P_dis. Set to 0 to disable (in which case prefer MultiHouseModel directly).
    """

    dt: float
    allow_export: bool = True
    cyclic_soc: bool = True
    lambda_deg: float = 0.0

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
        lambda_deg_by_house: Optional[Mapping[str, float]] = None,
    ) -> pyo.ConcreteModel:
        # 1) Build the base MILP — completely unchanged behaviour.
        base = MultiHouseModel(
            dt=self.dt,
            allow_export=self.allow_export,
            cyclic_soc=self.cyclic_soc,
        )
        m = base.make_instance(
            houses=houses,
            loads_by_house=loads_by_house,
            bat_params_by_house=bat_params_by_house,
            c_grid=c_grid,
            c_sell=c_sell,
            pv_by_house=pv_by_house,
            pv_total=pv_total,
            alpha_mode=alpha_mode,
        )

        # 2) Per-house lambda_deg (defaults to the scalar `self.lambda_deg`).
        lam = {
            h: float((lambda_deg_by_house or {}).get(h, self.lambda_deg))
            for h in list(m.H)
        }
        if all(v == 0.0 for v in lam.values()):
            # No effective penalty — return the base model untouched.
            return m

        m.lambda_deg = pyo.Param(m.H, initialize=lam, within=pyo.NonNegativeReals)

        # 3) Rewrite the objective: keep the original grid cost, add wear cost.
        original_expr = m.obj.expr
        m.del_component("obj")
        m.obj = pyo.Objective(
            expr=original_expr + sum(
                m.lambda_deg[h] * (m.P_ch[h, t] + m.P_dis[h, t]) * m.dt
                for h in m.H for t in m.T
            ),
            sense=pyo.minimize,
        )

        # 4) Track total throughput per house as a derived expression for
        # diagnostics; not part of the optimisation. Useful in meta.yaml.
        def _throughput_rule(mm, h):
            return sum((mm.P_ch[h, t] + mm.P_dis[h, t]) * mm.dt for t in mm.T)
        m.battery_throughput_kWh = pyo.Expression(m.H, rule=_throughput_rule)

        return m

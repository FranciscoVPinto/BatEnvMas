"""
Extensão do MultiHouseModel com penalidade linear de degradação da bateria.

O custo λ (€/kWh de throughput) é calculado por compute_degradation_cost_per_kwh()
ou definido directamente em model.battery_degradation_eur_per_kwh no YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import pyomo.environ as pyo

from .multi_house import MultiHouseModel


@dataclass
class MultiHouseModelDegradation:
    """
    Substituto de MultiHouseModel que adiciona ao objectivo:

        Σ_{h,t} λ[h] * (P_ch[h,t] + P_dis[h,t]) * dt

    Parâmetros
    ----------
    dt, allow_export, cyclic_soc  Passados ao MultiHouseModel.
    lambda_deg                    λ global (€/kWh). Calcular com
                                  compute_degradation_cost_per_kwh().
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
        """
        Constrói o modelo base e acrescenta a penalidade de degradação.

        lambda_deg_by_house  Override de λ por casa (opcional).
        """
        m = MultiHouseModel(
            dt=self.dt, allow_export=self.allow_export, cyclic_soc=self.cyclic_soc,
        ).make_instance(
            houses=houses, loads_by_house=loads_by_house,
            bat_params_by_house=bat_params_by_house,
            c_grid=c_grid, c_sell=c_sell,
            pv_by_house=pv_by_house, pv_total=pv_total,
            alpha_mode=alpha_mode,
        )

        # λ por casa (com override opcional)
        lam = {
            h: float((lambda_deg_by_house or {}).get(h, self.lambda_deg))
            for h in list(m.H)
        }
        if all(v == 0.0 for v in lam.values()):
            return m  # sem degradação: devolver modelo base sem alterações

        m.lambda_deg = pyo.Param(m.H, initialize=lam, within=pyo.NonNegativeReals)

        # Substituir objetivo: custo energia + custo degradação
        base_cost = m.obj.expr
        m.del_component("obj")
        m.obj = pyo.Objective(
            expr=base_cost + sum(
                m.lambda_deg[h] * (m.P_ch[h, t] + m.P_dis[h, t]) * m.dt
                for h in m.H for t in m.T
            ),
            sense=pyo.minimize,
        )

        # Throughput total por casa — útil para estimar ciclos consumidos
        m.battery_throughput_kWh = pyo.Expression(
            m.H,
            rule=lambda mm, h: sum(
                (mm.P_ch[h, t] + mm.P_dis[h, t]) * mm.dt for t in mm.T
            ),
        )
        return m

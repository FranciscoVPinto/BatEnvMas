"""
MILP com uma bateria independente por fracao (house-indexed).

Um unico binario y define o modo operacional de cada fracao:

  y = 0  ->  modo geracao   : PV >= carga, bateria pode CARREGAR (de PV), pode EXPORTAR
             P_imp = 0,  P_dis = 0
  y = 1  ->  modo consumo   : PV < carga, bateria pode DESCARREGAR, pode IMPORTAR
             P_exp = 0,  P_ch  = 0

A bateria NUNCA carrega da rede. O carregamento e sempre de excedente PV (y=0).
Consequencia: o fluxo PV e exacto e sem ambiguidade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pyomo.environ as pyo


# Escalaes oficiais de Potencia Contratada em Portugal (ERSE), em kVA.
PORTUGUESE_CONTRACTED_POWER_KVA = [
    1.15, 2.30, 3.45, 4.60, 5.75, 6.90,
    10.35, 13.80, 17.25, 20.70, 27.60, 34.50, 41.40,
]


def _load_dict(by_house, houses, T):
    """Converte {house: [v1...vT]} para {(house, t): vt} com t em [1, T]."""
    out = {}
    for h in houses:
        series = list(by_house[h])
        if len(series) != T:
            raise ValueError(f"Serie da casa '{h}' tem comprimento {len(series)}, esperado T={T}")
        for t in range(1, T + 1):
            out[(h, t)] = float(series[t - 1])
    return out


@dataclass
class MultiHouseModel:
    """
    MILP base: minimiza custo liquido de energia para uma comunidade de fracoes
    com baterias domesticas e PV partilhado.

    Parametros
    ----------
    dt : float           Duracao de cada intervalo (horas).
    allow_export : bool  Permite exportacao de excedente para a rede.
    cyclic_soc : bool    Impoe E[h,T] >= E_init[h] para horizonte finito.
    """

    dt: float
    allow_export: bool = True
    cyclic_soc: bool = True

    def make_instance(
        self,
        *,
        houses,
        loads_by_house,
        bat_params_by_house,
        c_grid,
        c_sell,
        pv_by_house=None,
        pv_total=None,
        alpha_mode="fixed",
    ):
        """
        Constroi o modelo Pyomo concreto.

        bat_params_by_house aceita as chaves:
          E_init, E_min, E_max, P_ch_max, P_dis_max, eta_ch, eta_dis
          P_contracted  -- Potencia Contratada (kW); aceita tambem P_grid_max.
        """
        if not houses:
            raise ValueError("houses nao pode ser vazio")
        if alpha_mode not in ("fixed", "optimal"):
            raise ValueError(f"alpha_mode deve ser 'fixed' ou 'optimal', obtido '{alpha_mode}'")

        houses = [str(h) for h in houses]
        T = len(next(iter(loads_by_house.values())))

        if alpha_mode == "fixed":
            if pv_by_house is None:
                raise ValueError("alpha_mode='fixed' requer pv_by_house")
            for h in houses:
                if len(loads_by_house[h]) != T or len(pv_by_house[h]) != T:
                    raise ValueError(f"Serie da casa '{h}' tem comprimento errado (esperado T={T})")
        else:
            if pv_total is None:
                raise ValueError("alpha_mode='optimal' requer pv_total")
            if len(pv_total) != T:
                raise ValueError(f"pv_total tem comprimento {len(pv_total)}, esperado T={T}")

        def expand_tariff(tariff):
            if isinstance(tariff, Mapping):
                return {h: list(tariff[h]) for h in houses}
            return {h: list(tariff) for h in houses}

        c_grid_h = expand_tariff(c_grid)
        c_sell_h = expand_tariff(c_sell)

        def bat(h, key, default):
            return float((bat_params_by_house.get(h) or {}).get(key, default))

        def contracted_power(h):
            bp = bat_params_by_house.get(h) or {}
            return float(bp.get("P_contracted", bp.get("P_grid_max", 1e9)))

        # --- Conjuntos ---
        m = pyo.ConcreteModel()
        m.H  = pyo.Set(initialize=houses, ordered=True)
        m.T  = pyo.RangeSet(1, T)
        m.TE = pyo.RangeSet(0, T)

        # --- Parametros ---
        m.dt        = pyo.Param(initialize=float(self.dt))
        m.kappa_exp = pyo.Param(initialize=int(self.allow_export), within=pyo.Binary)

        m.E_init       = pyo.Param(m.H, initialize=lambda mm, h: bat(h, "E_init",    0.0))
        m.E_min        = pyo.Param(m.H, initialize=lambda mm, h: bat(h, "E_min",     0.0))
        m.E_max        = pyo.Param(m.H, initialize=lambda mm, h: bat(h, "E_max",     0.0))
        m.P_ch_max     = pyo.Param(m.H, initialize=lambda mm, h: bat(h, "P_ch_max",  0.0))
        m.P_dis_max    = pyo.Param(m.H, initialize=lambda mm, h: bat(h, "P_dis_max", 0.0))
        m.eta_ch       = pyo.Param(m.H, initialize=lambda mm, h: bat(h, "eta_ch",    1.0))
        m.eta_dis      = pyo.Param(m.H, initialize=lambda mm, h: bat(h, "eta_dis",   1.0))
        m.P_contracted = pyo.Param(m.H, initialize=contracted_power, within=pyo.NonNegativeReals)

        m.Load   = pyo.Param(m.H, m.T, initialize=_load_dict(loads_by_house, houses, T),
                             within=pyo.NonNegativeReals)
        m.c_grid = pyo.Param(m.H, m.T, initialize=_load_dict(c_grid_h, houses, T),
                             within=pyo.NonNegativeReals)
        m.c_sell = pyo.Param(m.H, m.T, initialize=_load_dict(c_sell_h, houses, T),
                             within=pyo.NonNegativeReals)

        # --- PV: parametro (fixed) ou variavel de decisao (optimal) ---
        m.alpha_mode = pyo.Param(initialize=alpha_mode, within=pyo.Any)

        if alpha_mode == "fixed":
            m.PV = pyo.Param(m.H, m.T,
                             initialize=_load_dict(pv_by_house, houses, T),
                             within=pyo.NonNegativeReals)
        else:
            pv_total_dict = {t + 1: float(pv_total[t]) for t in range(T)}
            m.PV_total = pyo.Param(m.T, initialize=pv_total_dict, within=pyo.NonNegativeReals)
            m.PV = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)
            # Garante sum(alpha) = 1: todo o PV disponivel e alocado as fracoes
            m.pv_allocation = pyo.Constraint(
                m.T,
                rule=lambda mm, t: sum(mm.PV[h, t] for h in mm.H) == mm.PV_total[t],
            )
            m.pv_allocation_bound = pyo.Constraint(
                m.H, m.T,
                rule=lambda mm, h, t: mm.PV[h, t] <= mm.PV_total[t],
            )

        # --- Variaveis de decisao ---
        m.P_ch   = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)  # carga da bateria (kW)
        m.P_dis  = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)  # descarga da bateria (kW)
        m.P_imp  = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)  # importacao da rede (kW)
        m.P_exp  = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)  # exportacao para a rede (kW)
        m.P_curt = pyo.Var(m.H, m.T, within=pyo.NonNegativeReals)  # PV curtailed (kW)
        m.E      = pyo.Var(m.H, m.TE, within=pyo.NonNegativeReals) # energia na bateria (kWh)

        # Unico binario: y=0 modo geracao, y=1 modo consumo
        m.y = pyo.Var(m.H, m.T, within=pyo.Binary)

        # --- Dinamica de energia da bateria ---
        # E[t] = E[t-1] + eta_ch * P_ch * dt - P_dis / eta_dis * dt
        m.energy_balance = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: mm.E[h, t] == mm.E[h, t - 1] + (
                mm.eta_ch[h] * mm.P_ch[h, t]
                - (1.0 / mm.eta_dis[h]) * mm.P_dis[h, t]
            ) * mm.dt,
        )
        m.energy_init = pyo.Constraint(
            m.H, rule=lambda mm, h: mm.E[h, 0] == mm.E_init[h],
        )
        m.energy_bounds = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: pyo.inequality(mm.E_min[h], mm.E[h, t], mm.E_max[h]),
        )
        if self.cyclic_soc:
            m.energy_final = pyo.Constraint(
                m.H, rule=lambda mm, h: mm.E[h, T] >= mm.E_init[h],
            )

        # --- Rede: mutex importacao/exportacao, ambos limitados pela PC ---
        m.import_limit = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: mm.P_imp[h, t] <= mm.P_contracted[h] * mm.y[h, t],
        )
        m.export_limit = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: mm.P_exp[h, t] <= mm.P_contracted[h] * mm.kappa_exp * (1 - mm.y[h, t]),
        )

        # --- Bateria: carga no modo geracao, descarga no modo consumo ---
        #
        # charge_limit:    P_ch  <= P_ch_max  * (1 - y)
        #   y=0 (geracao)  -> P_ch  <= P_ch_max   [carrega de excedente PV]
        #   y=1 (consumo)  -> P_ch  = 0           [nao carrega — nunca da rede]
        #
        # discharge_limit: P_dis <= P_dis_max * y
        #   y=0 (geracao)  -> P_dis = 0           [nao descarrega — PV vai para carga/export]
        #   y=1 (consumo)  -> P_dis <= P_dis_max  [descarrega para reduzir importacao]
        m.charge_limit = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: mm.P_ch[h, t]  <= mm.P_ch_max[h]  * (1 - mm.y[h, t]),
        )
        m.discharge_limit = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: mm.P_dis[h, t] <= mm.P_dis_max[h] * mm.y[h, t],
        )

        # Curtailment limitado ao PV disponivel
        m.curtailment_limit = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: mm.P_curt[h, t] <= mm.PV[h, t],
        )

        # --- Balanco nodal de potencia ---
        # Load = (PV - P_curt) + P_dis - P_ch + P_imp - P_exp
        m.power_balance = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: mm.Load[h, t] == (
                mm.PV[h, t] - mm.P_curt[h, t]
                + mm.P_dis[h, t] - mm.P_ch[h, t]
                + mm.P_imp[h, t] - mm.P_exp[h, t]
            ),
        )

        # --- Fluxo exacto do PV (disponivel apos solve) ---
        #
        # Com charge_limit e discharge_limit, o fluxo e determinisitco:
        #
        #   y=0: PV_net = Load + P_ch + P_exp         (balanco sem P_dis e P_imp)
        #        P_pv_to_load = Load
        #        P_pv_to_bat  = P_ch
        #        P_pv_to_exp  = P_exp
        #
        #   y=1: PV_net = Load - P_dis - P_imp         (balanco sem P_ch e P_exp)
        #        P_pv_to_load = PV - P_curt = Load - P_dis - P_imp
        #        P_pv_to_bat  = 0
        #        P_pv_to_exp  = 0
        #
        # Invariante: P_pv_to_load + P_pv_to_bat + P_pv_to_exp + P_curt = PV
        m.P_pv_to_load = pyo.Expression(
            m.H, m.T,
            rule=lambda mm, h, t: mm.Load[h, t] - mm.P_dis[h, t] - mm.P_imp[h, t],
        )
        m.P_pv_to_bat = pyo.Expression(
            m.H, m.T,
            rule=lambda mm, h, t: mm.P_ch[h, t],
        )
        m.P_pv_to_exp = pyo.Expression(
            m.H, m.T,
            rule=lambda mm, h, t: mm.P_exp[h, t],
        )

        # --- Objectivo: minimizar custo liquido de energia ---
        m.obj = pyo.Objective(
            expr=sum(
                (m.c_grid[h, t] * m.P_imp[h, t] - m.c_sell[h, t] * m.P_exp[h, t]) * m.dt
                for h in m.H for t in m.T
            ),
            sense=pyo.minimize,
        )
        return m


# Alias de retrocompatibilidade
MultiHouseEnergySharingModel = MultiHouseModel

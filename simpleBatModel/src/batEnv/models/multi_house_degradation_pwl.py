"""
MultiHouseModelDegradationPWL — modelo de degradação por peças lineares (PWL)
dependente do estado de carga (SoC).

Fundamento físico
-----------------
A degradação por ciclagem de baterias Li-Ion / LFP não é uniforme ao longo do
SoC. Três regimes são identificados na literatura:

  - SoC baixo  (< s_lo): stress de descarga profunda — maior λ
      → risco de dissolução de cobre no anódo (Cu dissolution)
  - SoC médio  [s_lo, s_hi]: zona de operação preferencial — menor λ
      → ciclagem com menor sobretensão, menor crescimento de SEI
  - SoC alto   (> s_hi): stress de carga em alta tensão — λ intermédio
      → risco de deposição de lítio (lithium plating) à superfície do anódo

Comparação com o modelo linear simples (MultiHouseModelDegradation)
--------------------------------------------------------------------
  Simples:  custo_deg = λ_deg · (P_ch + P_dis) · dt
  PWL:      custo_deg = Σ_k λ[k] · z[k] · (P_ch + P_dis) · dt
            onde z[k] selecciona o bin de SoC corrente → λ varia com SoC.

Formulação MILP
---------------
Para cada (casa h, timestep t, bin k):

  1. Variável binária z[h,t,k] ∈ {0,1}
       Σ_k z[h,t,k] = 1   (exactamente um bin activo)

  2. Atribuição de bin por big-M (M_h = E_max[h] − E_min[h]):
       E[h,t] ≥ e_lo[h,k] − M_h · (1 − z[h,t,k])
       E[h,t] ≤ e_hi[h,k] + M_h · (1 − z[h,t,k])

  3. Linearização McCormick de q_ch = z · P_ch e q_dis = z · P_dis:
       q_ch[h,t,k] ≤ P_ch_max[h] · z[h,t,k]
       q_ch[h,t,k] ≤ P_ch[h,t]
       q_ch[h,t,k] ≥ P_ch[h,t] − P_ch_max[h] · (1 − z[h,t,k])
       q_ch[h,t,k] ≥ 0
       (análogo para q_dis)

  4. Custo de degradação adicionado ao objectivo:
       Σ_h Σ_t Σ_k  λ[h,k] · (q_ch[h,t,k] + q_dis[h,t,k]) · dt

Complexidade adicional (vs. modelo base)
-----------------------------------------
  K variáveis binárias   + 2K variáveis contínuas por (h,t)
  6K + 1 restrições por (h,t)
  Para K=3, T=96, H=5: +1 440 binárias e +2 880 contínuas.

Referências
-----------
  Schmalstieg, J. et al. (2014). A holistic aging model for Li(NiMnCo)O2
      based 18650 lithium-ion batteries. J. Power Sources, 257, 325–334.
  Xu, B. et al. (2018). Factoring the Cycle Aging Cost of Batteries
      Participating in Electricity Markets. IEEE Trans. Power Syst., 33(2).
  Koller, M. et al. (2015). Defining a degradation cost function for optimal
      control of a battery energy storage system. IEEE Grenoble PowerTech.
  Hesse, H.C. et al. (2017). Lithium-Ion Battery Storage for the Grid.
      Energies, 10(12), 2107.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pyomo.environ as pyo

from .multi_house import MultiHouseModel


# Valores padrão alinhados com Schmalstieg et al. (2014) e Hesse et al. (2017)
# para química LFP à temperatura ambiente:
#   bin 0: SoC ∈ [0 %,  20 %] — stress de descarga profunda → λ = 0.08 EUR/kWh
#   bin 1: SoC ∈ [20 %, 80 %] — operação preferencial        → λ = 0.03 EUR/kWh
#   bin 2: SoC ∈ [80 %,100 %] — stress de sobrecarga         → λ = 0.06 EUR/kWh
_DEFAULT_SOC_BREAKPOINTS: List[float] = [0.0, 0.2, 0.8, 1.0]
_DEFAULT_LAMBDA_BY_BIN: List[float] = [0.08, 0.03, 0.06]


def _validate_pwl_params(
    soc_breakpoints: List[float],
    lambda_by_bin: List[float],
    label: str = "",
) -> None:
    """Valida a consistência dos parâmetros PWL e lança ValueError se inválidos."""
    K = len(lambda_by_bin)
    prefix = f"[{label}] " if label else ""

    if K < 1:
        raise ValueError(f"{prefix}lambda_by_bin deve ter pelo menos 1 elemento.")
    if len(soc_breakpoints) != K + 1:
        raise ValueError(
            f"{prefix}soc_breakpoints deve ter len(lambda_by_bin)+1 = {K + 1} "
            f"elementos, mas tem {len(soc_breakpoints)}."
        )
    if abs(soc_breakpoints[0]) > 1e-9:
        raise ValueError(f"{prefix}soc_breakpoints[0] deve ser 0.0.")
    if abs(soc_breakpoints[-1] - 1.0) > 1e-9:
        raise ValueError(f"{prefix}soc_breakpoints[-1] deve ser 1.0.")
    for i in range(len(soc_breakpoints) - 1):
        if soc_breakpoints[i + 1] <= soc_breakpoints[i]:
            raise ValueError(
                f"{prefix}soc_breakpoints deve ser estritamente crescente. "
                f"Violado no índice {i}: {soc_breakpoints[i]} >= {soc_breakpoints[i + 1]}."
            )
    for k, lam in enumerate(lambda_by_bin):
        if lam < 0:
            raise ValueError(
                f"{prefix}lambda_by_bin[{k}] = {lam} deve ser >= 0."
            )


@dataclass
class MultiHouseModelDegradationPWL:
    """
    Substituto drop-in de MultiHouseModel com custo de degradação PWL
    dependente do SoC.

    Parâmetros
    ----------
    dt : float
        Duração do timestep em horas.
    allow_export : bool
        Permite exportação para a rede.
    cyclic_soc : bool
        Impõe E[h, T] >= E_init[h] (previne end-of-horizon cheating).
    soc_breakpoints : list[float]
        K+1 valores em [0, 1], estritamente crescentes, a definir os limites dos
        bins de SoC normalizado.  Padrão: [0.0, 0.2, 0.8, 1.0].
    lambda_by_bin : list[float]
        K valores λ (EUR/kWh) — um por bin.  O bin k aplica-se quando
        SoC ∈ [soc_breakpoints[k], soc_breakpoints[k+1]].
        Padrão: [0.08, 0.03, 0.06]  (calibrado para LFP, Schmalstieg 2014).
    lambda_by_bin_per_house : dict[str, list[float]], opcional
        Sobrepõe lambda_by_bin para casas específicas (e.g. baterias de química
        diferente).  Exemplo: {"casa_A": [0.07, 0.025, 0.055]}.
    """

    dt: float
    allow_export: bool = True
    cyclic_soc: bool = True
    soc_breakpoints: List[float] = field(
        default_factory=lambda: list(_DEFAULT_SOC_BREAKPOINTS)
    )
    lambda_by_bin: List[float] = field(
        default_factory=lambda: list(_DEFAULT_LAMBDA_BY_BIN)
    )
    lambda_by_bin_per_house: Optional[Dict[str, List[float]]] = None

    def __post_init__(self) -> None:
        _validate_pwl_params(self.soc_breakpoints, self.lambda_by_bin, "global")
        if self.lambda_by_bin_per_house:
            for h, lam_h in self.lambda_by_bin_per_house.items():
                _validate_pwl_params(self.soc_breakpoints, lam_h, f"house={h}")

    # ------------------------------------------------------------------
    # make_instance
    # ------------------------------------------------------------------

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
        """
        Constrói o modelo Pyomo MILP com degradação PWL por bins de SoC.

        A assinatura é idêntica à de MultiHouseModel.make_instance para
        compatibilidade de drop-in no pipeline run_case.py.
        """

        # 1) Modelo base: todas as variáveis e restrições originais ───────
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

        T = max(m.T)
        K = len(self.lambda_by_bin)
        bkpts = self.soc_breakpoints  # K+1 valores em [0, 1]

        # 2) Parâmetros físicos por casa ──────────────────────────────────
        def _p(h: str, key: str, default: float) -> float:
            return float((bat_params_by_house.get(h) or {}).get(key, default))

        e_min: Dict[str, float] = {h: _p(h, "E_min", 0.0) for h in houses}
        e_max: Dict[str, float] = {h: _p(h, "E_max", 0.0) for h in houses}

        # λ[h][k]: custo de degradação EUR/kWh no bin k da casa h
        lam: Dict[str, List[float]] = {}
        for h in houses:
            if self.lambda_by_bin_per_house and h in self.lambda_by_bin_per_house:
                lam[h] = list(self.lambda_by_bin_per_house[h])
            else:
                lam[h] = list(self.lambda_by_bin)

        # Casas sem bateria efectiva → degradação nula (sem alterar modelo)
        for h in houses:
            if e_max[h] <= e_min[h] + 1e-9:
                lam[h] = [0.0] * K

        # Optimização: se todos os λ são zero, devolver modelo base intacto
        if all(v == 0.0 for h in houses for v in lam[h]):
            return m

        # 3) Limites de energia por bin (energia absoluta, kWh) ───────────
        # Bin k activo quando SoC ∈ [bkpts[k], bkpts[k+1]]
        # → em energia: E ∈ [E_min + bkpts[k]*span, E_min + bkpts[k+1]*span]
        e_lo: Dict[str, List[float]] = {}
        e_hi: Dict[str, List[float]] = {}
        for h in houses:
            span = max(e_max[h] - e_min[h], 1e-9)
            e_lo[h] = [e_min[h] + bkpts[k] * span for k in range(K)]
            e_hi[h] = [e_min[h] + bkpts[k + 1] * span for k in range(K)]

        # 4) Sets e parâmetros adicionados ao modelo Pyomo ────────────────
        m.K = pyo.RangeSet(0, K - 1)

        # λ[h, k] — inicializado a partir do dict Python acima
        m.lambda_pwl = pyo.Param(
            m.H, m.K,
            initialize={(h, k): lam[h][k] for h in houses for k in range(K)},
            within=pyo.NonNegativeReals,
        )

        # Limites de energia por bin [kWh]
        m.e_bin_lo = pyo.Param(
            m.H, m.K,
            initialize={(h, k): e_lo[h][k] for h in houses for k in range(K)},
            within=pyo.Reals,
        )
        m.e_bin_hi = pyo.Param(
            m.H, m.K,
            initialize={(h, k): e_hi[h][k] for h in houses for k in range(K)},
            within=pyo.Reals,
        )

        # Big-M por casa = amplitude total do SoC (tight bound)
        m.M_soc = pyo.Param(
            m.H,
            initialize={h: max(e_max[h] - e_min[h], 1e-9) for h in houses},
            within=pyo.NonNegativeReals,
        )

        # 5) Variáveis de bin e auxiliares McCormick ──────────────────────

        # z[h, t, k] = 1 sse o SoC de h no timestep t pertence ao bin k
        m.z = pyo.Var(m.H, m.T, m.K, within=pyo.Binary)

        # q_ch[h,t,k] ≈ z[h,t,k] · P_ch[h,t]  (McCormick linearization)
        m.q_ch = pyo.Var(m.H, m.T, m.K, within=pyo.NonNegativeReals)

        # q_dis[h,t,k] ≈ z[h,t,k] · P_dis[h,t]
        m.q_dis = pyo.Var(m.H, m.T, m.K, within=pyo.NonNegativeReals)

        # 6) Exactamente um bin activo por (h, t) ─────────────────────────
        m.bin_unique = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: sum(mm.z[h, t, k] for k in mm.K) == 1,
        )

        # 7) Atribuição de bin: big-M para enforçar E ∈ [e_lo, e_hi] ─────
        # Quando z[h,t,k] = 1:  e_lo[h,k] ≤ E[h,t] ≤ e_hi[h,k]
        # Quando z[h,t,k] = 0:  restrições trivialmente satisfeitas por M_h

        def bin_lo_rule(mm, h, t, k):
            return mm.E[h, t] >= mm.e_bin_lo[h, k] - mm.M_soc[h] * (1 - mm.z[h, t, k])

        def bin_hi_rule(mm, h, t, k):
            return mm.E[h, t] <= mm.e_bin_hi[h, k] + mm.M_soc[h] * (1 - mm.z[h, t, k])

        m.bin_lo = pyo.Constraint(m.H, m.T, m.K, rule=bin_lo_rule)
        m.bin_hi = pyo.Constraint(m.H, m.T, m.K, rule=bin_hi_rule)

        # 8) Linearização McCormick para q_ch = z · P_ch ─────────────────
        # As quatro desigualdades garantem:
        #   z=1 → q_ch = P_ch  (limite inferior e superior coincidem)
        #   z=0 → q_ch = 0     (upper bound = 0; lower bound ≤ 0, trivial)

        def mc_ch_ub_bin(mm, h, t, k):
            # q ≤ P_ch_max · z  → quando z=0 força q=0
            return mm.q_ch[h, t, k] <= mm.P_ch_max[h] * mm.z[h, t, k]

        def mc_ch_ub_pow(mm, h, t, k):
            # q ≤ P_ch  → q não pode exceder a potência real
            return mm.q_ch[h, t, k] <= mm.P_ch[h, t]

        def mc_ch_lb(mm, h, t, k):
            # q ≥ P_ch − P_ch_max·(1−z)  → quando z=1 força q ≥ P_ch
            return mm.q_ch[h, t, k] >= mm.P_ch[h, t] - mm.P_ch_max[h] * (1 - mm.z[h, t, k])

        m.mc_ch_ub_bin = pyo.Constraint(m.H, m.T, m.K, rule=mc_ch_ub_bin)
        m.mc_ch_ub_pow = pyo.Constraint(m.H, m.T, m.K, rule=mc_ch_ub_pow)
        m.mc_ch_lb = pyo.Constraint(m.H, m.T, m.K, rule=mc_ch_lb)

        # 9) Linearização McCormick para q_dis = z · P_dis ────────────────

        def mc_dis_ub_bin(mm, h, t, k):
            return mm.q_dis[h, t, k] <= mm.P_dis_max[h] * mm.z[h, t, k]

        def mc_dis_ub_pow(mm, h, t, k):
            return mm.q_dis[h, t, k] <= mm.P_dis[h, t]

        def mc_dis_lb(mm, h, t, k):
            return mm.q_dis[h, t, k] >= mm.P_dis[h, t] - mm.P_dis_max[h] * (1 - mm.z[h, t, k])

        m.mc_dis_ub_bin = pyo.Constraint(m.H, m.T, m.K, rule=mc_dis_ub_bin)
        m.mc_dis_ub_pow = pyo.Constraint(m.H, m.T, m.K, rule=mc_dis_ub_pow)
        m.mc_dis_lb = pyo.Constraint(m.H, m.T, m.K, rule=mc_dis_lb)

        # 10) Reescrever objectivo com custo de degradação PWL ─────────────
        original_expr = m.obj.expr
        m.del_component("obj")
        m.obj = pyo.Objective(
            expr=original_expr + sum(
                m.lambda_pwl[h, k] * (m.q_ch[h, t, k] + m.q_dis[h, t, k]) * m.dt
                for h in m.H for t in m.T for k in m.K
            ),
            sense=pyo.minimize,
        )

        # 11) Expressões diagnósticas (não entram na optimização) ──────────

        def _deg_cost_rule(mm, h):
            """Custo total de degradação PWL para a casa h [EUR]."""
            return sum(
                mm.lambda_pwl[h, k] * (mm.q_ch[h, t, k] + mm.q_dis[h, t, k]) * mm.dt
                for t in mm.T for k in mm.K
            )

        m.pwl_degradation_cost_EUR = pyo.Expression(m.H, rule=_deg_cost_rule)

        def _throughput_rule(mm, h):
            """Throughput total da bateria da casa h [kWh]."""
            return sum(
                (mm.q_ch[h, t, k] + mm.q_dis[h, t, k]) * mm.dt
                for t in mm.T for k in mm.K
            )

        m.battery_throughput_kWh = pyo.Expression(m.H, rule=_throughput_rule)

        def _bin_hours_rule(mm, h, k):
            """Número de timesteps em que a casa h operou no bin k."""
            return sum(mm.z[h, t, k] for t in mm.T)

        m.bin_hours = pyo.Expression(m.H, m.K, rule=_bin_hours_rule)

        return m

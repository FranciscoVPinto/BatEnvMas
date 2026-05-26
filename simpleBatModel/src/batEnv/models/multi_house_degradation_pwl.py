from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pyomo.environ as pyo

from .multi_house import MultiHouseModel


_DEFAULT_SOC_BREAKPOINTS: List[float] = [0.0, 0.2, 0.8, 1.0]
_DEFAULT_LAMBDA_BY_BIN: List[float] = [0.08, 0.03, 0.06]


def _validate_pwl_params(
    soc_breakpoints: List[float],
    lambda_by_bin: List[float],
    label: str = "",
) -> None:
    K = len(lambda_by_bin)
    prefix = f"[{label}] " if label else ""
    if K < 1:
        raise ValueError(f"{prefix}lambda_by_bin must have at least 1 element.")
    if len(soc_breakpoints) != K + 1:
        raise ValueError(
            f"{prefix}soc_breakpoints must have len(lambda_by_bin)+1 = {K + 1} "
            f"elements, got {len(soc_breakpoints)}."
        )
    if abs(soc_breakpoints[0]) > 1e-9:
        raise ValueError(f"{prefix}soc_breakpoints[0] must be 0.0.")
    if abs(soc_breakpoints[-1] - 1.0) > 1e-9:
        raise ValueError(f"{prefix}soc_breakpoints[-1] must be 1.0.")
    for i in range(len(soc_breakpoints) - 1):
        if soc_breakpoints[i + 1] <= soc_breakpoints[i]:
            raise ValueError(
                f"{prefix}soc_breakpoints must be strictly increasing. "
                f"Violated at index {i}: {soc_breakpoints[i]} >= {soc_breakpoints[i + 1]}."
            )
    for k, lam in enumerate(lambda_by_bin):
        if lam < 0:
            raise ValueError(f"{prefix}lambda_by_bin[{k}] = {lam} must be >= 0.")


@dataclass
class MultiHouseModelDegradationPWL:
    """
    Drop-in replacement for MultiHouseModel with SoC-dependent (PWL) battery
    degradation cost. Splits the SoC range into K bins, each with its own
    degradation cost lambda[k] (EUR/kWh). Bin assignment uses binary variables
    with big-M constraints; bilinear products (z * P_ch/dis) are linearised
    via McCormick envelopes.

    Parameters
    ----------
    dt : float
        Timestep duration in hours.
    allow_export : bool
        Whether grid export is allowed.
    cyclic_soc : bool
        Enforce E[h, T] >= E_init[h] to prevent end-of-horizon cheating.
    soc_breakpoints : list[float]
        K+1 values in [0, 1], strictly increasing. Defaults: [0.0, 0.2, 0.8, 1.0].
    lambda_by_bin : list[float]
        K degradation costs (EUR/kWh), one per bin. Defaults: [0.08, 0.03, 0.06].
    lambda_by_bin_per_house : dict[str, list[float]], optional
        Per-house override for lambda_by_bin.
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

        K = len(self.lambda_by_bin)
        bkpts = self.soc_breakpoints

        def _p(h: str, key: str, default: float) -> float:
            return float((bat_params_by_house.get(h) or {}).get(key, default))

        e_min: Dict[str, float] = {h: _p(h, "E_min", 0.0) for h in houses}
        e_max: Dict[str, float] = {h: _p(h, "E_max", 0.0) for h in houses}

        lam: Dict[str, List[float]] = {}
        for h in houses:
            if self.lambda_by_bin_per_house and h in self.lambda_by_bin_per_house:
                lam[h] = list(self.lambda_by_bin_per_house[h])
            else:
                lam[h] = list(self.lambda_by_bin)

        for h in houses:
            if e_max[h] <= e_min[h] + 1e-9:
                lam[h] = [0.0] * K

        if all(v == 0.0 for h in houses for v in lam[h]):
            return m

        # Energy bounds per bin (absolute kWh)
        e_lo: Dict[str, List[float]] = {}
        e_hi: Dict[str, List[float]] = {}
        for h in houses:
            span = max(e_max[h] - e_min[h], 1e-9)
            e_lo[h] = [e_min[h] + bkpts[k] * span for k in range(K)]
            e_hi[h] = [e_min[h] + bkpts[k + 1] * span for k in range(K)]

        m.K = pyo.RangeSet(0, K - 1)

        m.lambda_pwl = pyo.Param(
            m.H, m.K,
            initialize={(h, k): lam[h][k] for h in houses for k in range(K)},
            within=pyo.NonNegativeReals,
        )
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
        m.M_soc = pyo.Param(
            m.H,
            initialize={h: max(e_max[h] - e_min[h], 1e-9) for h in houses},
            within=pyo.NonNegativeReals,
        )

        # z[h,t,k] = 1 iff SoC of h at t is in bin k
        m.z = pyo.Var(m.H, m.T, m.K, within=pyo.Binary)
        # McCormick linearisations: q_ch ≈ z * P_ch, q_dis ≈ z * P_dis
        m.q_ch = pyo.Var(m.H, m.T, m.K, within=pyo.NonNegativeReals)
        m.q_dis = pyo.Var(m.H, m.T, m.K, within=pyo.NonNegativeReals)

        m.bin_unique = pyo.Constraint(
            m.H, m.T,
            rule=lambda mm, h, t: sum(mm.z[h, t, k] for k in mm.K) == 1,
        )
        m.bin_lo = pyo.Constraint(
            m.H, m.T, m.K,
            rule=lambda mm, h, t, k: mm.E[h, t] >= mm.e_bin_lo[h, k] - mm.M_soc[h] * (1 - mm.z[h, t, k]),
        )
        m.bin_hi = pyo.Constraint(
            m.H, m.T, m.K,
            rule=lambda mm, h, t, k: mm.E[h, t] <= mm.e_bin_hi[h, k] + mm.M_soc[h] * (1 - mm.z[h, t, k]),
        )
        m.mc_ch_ub_bin = pyo.Constraint(
            m.H, m.T, m.K,
            rule=lambda mm, h, t, k: mm.q_ch[h, t, k] <= mm.P_ch_max[h] * mm.z[h, t, k],
        )
        m.mc_ch_ub_pow = pyo.Constraint(
            m.H, m.T, m.K,
            rule=lambda mm, h, t, k: mm.q_ch[h, t, k] <= mm.P_ch[h, t],
        )
        m.mc_ch_lb = pyo.Constraint(
            m.H, m.T, m.K,
            rule=lambda mm, h, t, k: mm.q_ch[h, t, k] >= mm.P_ch[h, t] - mm.P_ch_max[h] * (1 - mm.z[h, t, k]),
        )
        m.mc_dis_ub_bin = pyo.Constraint(
            m.H, m.T, m.K,
            rule=lambda mm, h, t, k: mm.q_dis[h, t, k] <= mm.P_dis_max[h] * mm.z[h, t, k],
        )
        m.mc_dis_ub_pow = pyo.Constraint(
            m.H, m.T, m.K,
            rule=lambda mm, h, t, k: mm.q_dis[h, t, k] <= mm.P_dis[h, t],
        )
        m.mc_dis_lb = pyo.Constraint(
            m.H, m.T, m.K,
            rule=lambda mm, h, t, k: mm.q_dis[h, t, k] >= mm.P_dis[h, t] - mm.P_dis_max[h] * (1 - mm.z[h, t, k]),
        )

        original_expr = m.obj.expr
        m.del_component("obj")
        m.obj = pyo.Objective(
            expr=original_expr + sum(
                m.lambda_pwl[h, k] * (m.q_ch[h, t, k] + m.q_dis[h, t, k]) * m.dt
                for h in m.H for t in m.T for k in m.K
            ),
            sense=pyo.minimize,
        )

        m.pwl_degradation_cost_EUR = pyo.Expression(
            m.H,
            rule=lambda mm, h: sum(
                mm.lambda_pwl[h, k] * (mm.q_ch[h, t, k] + mm.q_dis[h, t, k]) * mm.dt
                for t in mm.T for k in mm.K
            ),
        )
        m.battery_throughput_kWh = pyo.Expression(
            m.H,
            rule=lambda mm, h: sum(
                (mm.q_ch[h, t, k] + mm.q_dis[h, t, k]) * mm.dt
                for t in mm.T for k in mm.K
            ),
        )
        m.bin_hours = pyo.Expression(
            m.H, m.K,
            rule=lambda mm, h, k: sum(mm.z[h, t, k] for t in mm.T),
        )

        return m

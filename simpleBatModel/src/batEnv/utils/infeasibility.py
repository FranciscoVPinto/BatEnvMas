from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass
class Violation:
    house: str
    t: int  # 0-indexed
    load: float
    pv: float
    max_supply: float
    missing: float
    reason: str


def _top_n(violations: list[Violation], n: int) -> list[Violation]:
    return sorted(violations, key=lambda v: v.missing, reverse=True)[: max(1, n)]


def check_single_house_power_feasibility(
    *,
    load: Sequence[float],
    pv: Sequence[float],
    dt_hours: float,
    P_grid_max: float,
    P_dis_max: float,
    E_init: float,
    E_max: float,
    eta_dis: float,
    house: str = "H1",
    max_rows: int = 20,
) -> list[Violation]:
    """
    Fast sufficient infeasibility checks for the single-house model.

    If Load[t] exceeds PV[t] + P_grid_max + max_discharge_possible, the MILP is infeasible.
    max_discharge_possible is limited by:
      - inverter: P_dis_max
      - energy: E_prev * eta_dis / dt_hours (t=0 uses E_init, later uses E_max as best case)
    """
    T = len(load)
    if len(pv) != T:
        raise ValueError("load/pv length mismatch")

    violations: list[Violation] = []
    for t in range(T):
        E_best = E_init if t == 0 else E_max
        max_dis_energy = (E_best * eta_dis / dt_hours) if dt_hours > 0 else 0.0
        max_dis = min(P_dis_max, max_dis_energy)

        max_supply = float(pv[t]) + float(P_grid_max) + float(max_dis)
        missing = float(load[t]) - max_supply
        if missing > 1e-6:
            reason = (
                "Load exceeds PV + grid import limit + max battery discharge.\n"
                "Constraints involved: grid import limit (P_imp<=P_grid_max), "
                "battery discharge limit (P_dis<=P_dis_max) and SoC bounds (E>=E_min)."
            )
            violations.append(
                Violation(house=house, t=t, load=float(load[t]), pv=float(pv[t]), max_supply=max_supply, missing=missing, reason=reason)
            )

    return _top_n(violations, max_rows) if violations else []


def check_multi_house_power_feasibility(
    *,
    loads_by_house: Mapping[str, Sequence[float]],
    pv_by_house: Mapping[str, Sequence[float]],
    dt_hours: float,
    bat_params_by_house: Mapping[str, Mapping],
    max_rows: int = 30,
) -> list[Violation]:
    """
    Fast infeasibility checks for the joint multi-house model.

    Per-house sufficient condition:
      Load[h,t] <= PV[h,t] + P_grid_max[h] + max_discharge[h,t]

    System-level sufficient condition:
      sum_h Load[h,t] <= sum_h (PV[h,t] + P_grid_max[h] + max_discharge_best[h,t])
    """
    houses = list(loads_by_house.keys())
    if not houses:
        return []

    T = len(next(iter(loads_by_house.values())))
    violations: list[Violation] = []

    # Precompute best-case discharge per house/time
    best_dis: dict[str, list[float]] = {}
    for h in houses:
        p = bat_params_by_house.get(h, {})
        P_dis_max = float(p.get("P_dis_max", 0.0))
        E_init = float(p.get("E_init", 0.0))
        E_max = float(p.get("E_max", 0.0))
        eta_dis = float(p.get("eta_dis", 1.0))
        arr = []
        for t in range(T):
            E_best = E_init if t == 0 else E_max
            max_dis_energy = (E_best * eta_dis / dt_hours) if dt_hours > 0 else 0.0
            arr.append(min(P_dis_max, max_dis_energy))
        best_dis[h] = arr

    # Per-house check
    for h in houses:
        load = loads_by_house[h]
        pv = pv_by_house[h]
        p = bat_params_by_house.get(h, {})
        P_grid_max = float(p.get("P_grid_max", 0.0))

        for t in range(T):
            max_supply = float(pv[t]) + P_grid_max + best_dis[h][t]
            missing = float(load[t]) - max_supply
            if missing > 1e-6:
                reason = (
                    "House load exceeds PV + grid import limit + max battery discharge.\n"
                    "Constraints involved: grid import limit, discharge limit and SoC bounds."
                )
                violations.append(
                    Violation(house=h, t=t, load=float(load[t]), pv=float(pv[t]), max_supply=max_supply, missing=missing, reason=reason)
                )

    # System-level check
    for t in range(T):
        total_load = sum(float(loads_by_house[h][t]) for h in houses)
        total_supply = 0.0
        for h in houses:
            p = bat_params_by_house.get(h, {})
            P_grid_max = float(p.get("P_grid_max", 0.0))
            total_supply += float(pv_by_house[h][t]) + P_grid_max + best_dis[h][t]

        missing = total_load - total_supply
        if missing > 1e-6:
            reason = (
                "System total load exceeds total PV + total grid import limits + total max battery discharge.\n"
                "Even with sharing, community balance prevents creating net power.\n"
                "Constraints involved: per-house grid import limits and discharge limits + community balance."
            )
            violations.append(
                Violation(house="__SYSTEM__", t=t, load=total_load, pv=sum(float(pv_by_house[h][t]) for h in houses),
                          max_supply=total_supply, missing=missing, reason=reason)
            )

    return _top_n(violations, max_rows) if violations else []


def format_violations(violations: Iterable[Violation]) -> str:
    lines = []
    for v in violations:
        lines.append(
            f"- {v.house} @ t={v.t}: Load={v.load:.3f} kW, PV={v.pv:.3f} kW, "
            f"MaxSupply≈{v.max_supply:.3f} kW -> Missing={v.missing:.3f} kW"
        )
    if lines:
        lines.append("")
        lines.append("Details (why):")
        # show the reason from the worst one
        worst = next(iter(violations))
        lines.append(worst.reason)
    return "\n".join(lines)
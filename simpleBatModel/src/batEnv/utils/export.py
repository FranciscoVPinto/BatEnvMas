from __future__ import annotations

from typing import Dict

import pandas as pd
import pyomo.environ as pyo


_PYOMO_VAL_ERRORS = (ValueError, AttributeError, KeyError)


def _val(x) -> float:
    """Extrai valor de uma variavel/expressao Pyomo; devolve 0.0 se indisponivel."""
    import logging
    _log = logging.getLogger("pyomo.core")
    _old = _log.level
    _log.setLevel(logging.CRITICAL)
    try:
        v = pyo.value(x, exception=False)
        return float(v) if v is not None else 0.0
    except _PYOMO_VAL_ERRORS:
        return 0.0
    finally:
        _log.setLevel(_old)


def _extract_house_dataframe(m, house_id: str) -> pd.DataFrame:
    T = list(m.T)
    E = []
    for t in T:
        try:
            E.append(_val(m.E[house_id, t]))
        except KeyError:
            E.append(0.0)

    df = pd.DataFrame({
        "t":      [int(t) for t in T],
        "Load":   [_val(m.Load[house_id, t])   for t in T],
        "PV":     [_val(m.PV[house_id, t])     for t in T],
        "P_imp":  [_val(m.P_imp[house_id, t])  for t in T],
        "P_exp":  [_val(m.P_exp[house_id, t])  for t in T],
        "P_ch":   [_val(m.P_ch[house_id, t])   for t in T],
        "P_dis":  [_val(m.P_dis[house_id, t])  for t in T],
        "P_curt": [_val(m.P_curt[house_id, t]) for t in T],
        "E":      E,
        "c_grid": [_val(m.c_grid[house_id, t]) for t in T],
        "c_sell": [_val(m.c_sell[house_id, t]) for t in T],
    })

    # Fluxo exacto do PV -- disponivel quando o modelo tem a restricao no_grid_charging.
    # Invariante: P_pv_to_load + P_pv_to_bat + P_pv_to_exp + P_curt = PV
    if hasattr(m, "P_pv_to_load"):
        df["P_pv_to_load"] = [max(0.0, _val(m.P_pv_to_load[house_id, t])) for t in T]
        df["P_pv_to_bat"]  = [_val(m.P_pv_to_bat[house_id, t])            for t in T]
        df["P_pv_to_exp"]  = [_val(m.P_pv_to_exp[house_id, t])            for t in T]

    return df


def multi_model_to_dataframes(m) -> Dict[str, pd.DataFrame]:
    """Exporta o modelo unificado como {house_id: dataframe}."""
    return {str(h): _extract_house_dataframe(m, h) for h in list(m.H)}


def model_to_dataframe(m, house_id=None) -> pd.DataFrame:
    """Compatibilidade: exporta uma unica casa (primeira se house_id=None)."""
    houses = list(getattr(m, "H", []))
    if not houses:
        raise ValueError("Expected a unified house-indexed model with set H.")
    chosen = str(house_id) if house_id is not None else str(houses[0])
    if chosen not in [str(h) for h in houses]:
        raise KeyError(f"House '{chosen}' not found in model.")
    return _extract_house_dataframe(m, chosen)


def extract_pwl_metrics_dataframe(m) -> pd.DataFrame:
    """Extrai metricas PWL por casa de um modelo de degradacao PWL resolvido."""
    if not (hasattr(m, "K") and hasattr(m, "pwl_degradation_cost_EUR")):
        return pd.DataFrame()

    K = list(m.K)
    rows = []
    for h in list(m.H):
        row: dict = {"house": str(h)}
        try:
            row["pwl_degradation_cost_EUR"] = _val(m.pwl_degradation_cost_EUR[h])
        except (KeyError, AttributeError):
            row["pwl_degradation_cost_EUR"] = 0.0
        try:
            row["battery_throughput_kWh"] = _val(m.battery_throughput_kWh[h])
        except (KeyError, AttributeError):
            row["battery_throughput_kWh"] = 0.0
        for k in K:
            try:
                row[f"bin_hours_{k}"] = _val(m.bin_hours[h, k])
            except (KeyError, AttributeError):
                row[f"bin_hours_{k}"] = 0.0
        rows.append(row)

    return pd.DataFrame(rows)

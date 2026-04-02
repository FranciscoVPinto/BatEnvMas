from __future__ import annotations

from typing import Dict

import pandas as pd
import pyomo.environ as pyo


def _val(x) -> float:
    try:
        return float(pyo.value(x))
    except Exception:
        return 0.0


def _extract_house_dataframe(m, house_id: str) -> pd.DataFrame:
    T = list(m.T)
    E = []
    for t in T:
        try:
            E.append(_val(m.E[house_id, t]))
        except Exception:
            E.append(0.0)

    return pd.DataFrame(
        {
            "t": [int(t) for t in T],
            "Load": [_val(m.Load[house_id, t]) for t in T],
            "PV": [_val(m.PV[house_id, t]) for t in T],
            "P_imp": [_val(m.P_imp[house_id, t]) for t in T],
            "P_exp": [_val(m.P_exp[house_id, t]) for t in T],
            "P_ch": [_val(m.P_ch[house_id, t]) for t in T],
            "P_dis": [_val(m.P_dis[house_id, t]) for t in T],
            "P_curt": [_val(m.P_curt[house_id, t]) for t in T],
            "E": E,
            "c_grid": [_val(m.c_grid[house_id, t]) for t in T],
            "c_sell": [_val(m.c_sell[house_id, t]) for t in T],
        }
    )


def multi_model_to_dataframes(m) -> Dict[str, pd.DataFrame]:
    """
    Export the unified house-indexed model as {house_id: dataframe}.
    """
    return {str(h): _extract_house_dataframe(m, h) for h in list(m.H)}


def model_to_dataframe(m, house_id: str | None = None) -> pd.DataFrame:
    """
    Backward-compatible single-house export.

    When `house_id` is omitted, the first house in m.H is used.
    """
    houses = list(getattr(m, "H", []))
    if not houses:
        raise ValueError("Expected a unified house-indexed model with set H.")
    chosen = str(house_id) if house_id is not None else str(houses[0])
    if chosen not in [str(h) for h in houses]:
        raise KeyError(f"House '{chosen}' not found in model.")
    return _extract_house_dataframe(m, chosen)


def model_to_dataframes(m) -> Dict[str, pd.DataFrame]:
    return multi_model_to_dataframes(m)

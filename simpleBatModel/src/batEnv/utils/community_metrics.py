from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


COMMUNITY_ID = "_COMMUNITY"


def aggregate_community_timeseries(house_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate multiple house result CSVs into a synthetic community dataframe."""
    if not house_dfs:
        return pd.DataFrame()

    ref = next(iter(house_dfs.values())).copy()
    if "t" not in ref.columns:
        raise ValueError("Result CSVs must contain column 't'")

    out = pd.DataFrame({"t": ref["t"].astype(int)})
    sum_cols = ["Load", "PV", "P_imp", "P_exp", "P_ch", "P_dis", "P_curt", "E"]
    passthrough_cols = ["c_grid", "c_sell"]

    for col in sum_cols:
        out[col] = 0.0

    for _, df in house_dfs.items():
        df2 = df.copy().sort_values("t").reset_index(drop=True)
        for col in sum_cols:
            if col in df2.columns:
                out[col] += df2[col].astype(float).to_numpy()
        for col in passthrough_cols:
            if col in df2.columns and col not in out.columns:
                out[col] = df2[col].astype(float).to_numpy()

    for col in passthrough_cols:
        if col not in out.columns:
            out[col] = np.nan

    out["P_simul_imp_exp"] = np.minimum(out["P_imp"].to_numpy(), out["P_exp"].to_numpy())
    return out


def compute_community_extra_metrics(df_comm: pd.DataFrame, dt_hours: float) -> Dict[str, Any]:
    """
    Community-level metrics computed from the aggregated timeseries.

    Includes Self-Sufficiency / Self-Consumption at community level — these are
    the standard CER (Comunidade de Energia Renovavel) headline metrics.
    """
    if df_comm.empty:
        return {}

    def _E(col: str) -> float:
        if col not in df_comm.columns:
            return 0.0
        return float(df_comm[col].sum() * dt_hours)

    e_load = _E("Load")
    e_pv = _E("PV")
    e_imp = _E("P_imp")
    e_exp = _E("P_exp")
    e_curt = _E("P_curt")

    out: Dict[str, Any] = {
        "E_simul_imp_exp_kWh": float(df_comm["P_simul_imp_exp"].sum() * dt_hours) if "P_simul_imp_exp" in df_comm.columns else 0.0,
        "E_imp_kWh_COMM": e_imp,
        "E_exp_kWh_COMM": e_exp,
        "E_curt_kWh_COMM": e_curt,
    }

    e_ch = _E("P_ch")

    if e_pv > 0:
        out["Curt_frac_of_PV_COMM"] = float(e_curt / e_pv)
        sc = (e_pv - e_curt - e_exp) / e_pv
        out["Self_Consumption_COMM"] = float(max(0.0, min(1.0, sc)))

    e_surplus = e_ch + e_exp + e_curt
    if e_surplus > 0:
        out["PV_surplus_kWh_COMM"] = float(e_surplus)
        out["Surplus_captured_frac_COMM"] = float(
            max(0.0, min(1.0, e_ch / e_surplus))
        )

    if e_load > 0:
        ss = (e_load - e_imp) / e_load
        out["Self_Sufficiency_COMM"] = float(max(0.0, min(1.0, ss)))

    if e_imp > 0:
        out["Simul_frac_of_import"] = float(out.get("E_simul_imp_exp_kWh", 0.0) / e_imp)

    return out


# ---------- fairness across houses (within a single case) ----------

def _gini(values: Iterable[float]) -> float:
    """Gini coefficient over a non-negative sequence. Returns NaN for empty/all-zero."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return float("nan")
    arr = np.sort(arr)
    s = arr.sum()
    if s <= 0:
        return float("nan")
    n = arr.size
    # Standard formula: G = (2 * sum(i*x_i) - (n+1)*sum(x)) / (n * sum(x))
    cum = np.arange(1, n + 1) * arr
    return float((2.0 * cum.sum() - (n + 1) * s) / (n * s))


def compute_fairness_metrics(per_house_metrics: pd.DataFrame, *, metric: str = "Cost_total_EUR") -> Dict[str, Any]:
    """
    Equity metrics across houses for a single case.

    `per_house_metrics` is a DataFrame with one row per house and a column
    matching `metric`. Houses named `_COMMUNITY` (synthetic aggregate) are
    excluded so the dispersion reflects real apartments only.
    """
    if per_house_metrics.empty or metric not in per_house_metrics.columns:
        return {}

    df = per_house_metrics
    if "house" in df.index.names:
        df = df.reset_index()
    if "house" in df.columns:
        df = df[df["house"] != COMMUNITY_ID]

    vals = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
    if vals.size < 2:
        return {}

    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    out: Dict[str, Any] = {
        f"{metric}_mean_house": mean,
        f"{metric}_std_house": std,
        f"{metric}_min_house": float(vals.min()),
        f"{metric}_max_house": float(vals.max()),
        f"{metric}_range_house": float(vals.max() - vals.min()),
    }
    if mean != 0:
        out[f"{metric}_CV_house"] = float(std / abs(mean))
    out[f"{metric}_Gini_house"] = _gini(vals)
    return out

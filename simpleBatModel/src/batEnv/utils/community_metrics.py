from __future__ import annotations

from typing import Any, Dict

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
    if df_comm.empty:
        return {}

    def _E(col: str) -> float:
        if col not in df_comm.columns:
            return 0.0
        return float(df_comm[col].sum() * dt_hours)

    out: Dict[str, Any] = {
        "E_simul_imp_exp_kWh": float(df_comm["P_simul_imp_exp"].sum() * dt_hours) if "P_simul_imp_exp" in df_comm.columns else 0.0,
        "E_imp_kWh_COMM": _E("P_imp"),
        "E_exp_kWh_COMM": _E("P_exp"),
        "E_curt_kWh_COMM": _E("P_curt"),
    }

    if "PV" in df_comm.columns:
        Epv = _E("PV")
        if Epv > 0:
            out["Curt_frac_of_PV_COMM"] = float(out["E_curt_kWh_COMM"] / Epv)

    if out["E_imp_kWh_COMM"] > 0:
        out["Simul_frac_of_import"] = float(out.get("E_simul_imp_exp_kWh", 0.0) / out["E_imp_kWh_COMM"])

    return out

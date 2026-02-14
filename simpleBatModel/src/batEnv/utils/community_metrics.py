from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


COMMUNITY_ID = "_COMMUNITY"


def aggregate_community_timeseries(house_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Agrega vários results_house_*.csv num dataframe sintético da comunidade.

    Produz (pelo menos):
      t, Load, PV, P_imp, P_exp, P_ch, P_dis, E, c_grid, c_sell, P_simul_imp_exp

    Notas:
      - E é soma do armazenamento total (se existir E nos dfs).
      - Colunas em falta numa casa contam como 0.
      - Assume que todas as casas usam o mesmo índice temporal 't'.
    """
    if not house_dfs:
        return pd.DataFrame()

    ref = next(iter(house_dfs.values())).copy()
    if "t" not in ref.columns:
        raise ValueError("Result CSVs must contain column 't'")

    out = pd.DataFrame({"t": ref["t"].astype(int)})

    sum_cols = ["Load", "PV", "P_imp", "P_exp", "P_ch", "P_dis", "E"]
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

    # proxy de simultaneidade: importação e exportação ao mesmo tempo
    out["P_simul_imp_exp"] = np.minimum(out["P_imp"].to_numpy(), out["P_exp"].to_numpy())

    return out


def compute_community_extra_metrics(df_comm: pd.DataFrame, dt_hours: float) -> Dict[str, Any]:
    """
    KPIs extra (além dos que compute_summary_metrics já calcula).
    """
    if df_comm.empty:
        return {}

    out: Dict[str, Any] = {}

    if "P_simul_imp_exp" in df_comm.columns:
        out["E_simul_imp_exp_kWh"] = float(df_comm["P_simul_imp_exp"].sum() * dt_hours)

    if "P_imp" in df_comm.columns and "P_exp" in df_comm.columns:
        Eimp = float(df_comm["P_imp"].sum() * dt_hours)
        Eexp = float(df_comm["P_exp"].sum() * dt_hours)
        out["E_imp_kWh_COMM"] = Eimp
        out["E_exp_kWh_COMM"] = Eexp
        if Eimp > 0:
            out["Simul_frac_of_import"] = float(out.get("E_simul_imp_exp_kWh", 0.0) / Eimp)

    return out

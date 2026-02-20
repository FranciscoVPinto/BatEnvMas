from __future__ import annotations

from typing import Dict
import pandas as pd
import pyomo.environ as pyo


def _v(x):
    try:
        return float(pyo.value(x))
    except Exception:
        return float(x)


def model_to_dataframe(model: pyo.ConcreteModel) -> pd.DataFrame:
    """
    Export a single-house model to a dataframe.
    """
    kappa_exp = None
    if hasattr(model, "kappa_exp"):
        try:
            kappa_exp = int(round(_v(model.kappa_exp)))
        except Exception:
            kappa_exp = None

    rows = []
    for t in model.T:
        row = {
            "t": int(t),
            "Load": _v(model.Load[t]),
            "PV": _v(model.PV[t]),
            "c_grid": _v(model.c_grid[t]),
            "c_sell": _v(model.c_sell[t]),
            "P_imp": _v(model.P_imp[t]),
            "P_exp": _v(model.P_exp[t]),
            "P_ch": _v(model.P_ch[t]),
            "P_dis": _v(model.P_dis[t]),
            "E": _v(model.E[t]),
            "x": int(round(_v(model.x[t]))),
            "y": int(round(_v(model.y[t]))),
        }
        if hasattr(model, "P_curt"):
            row["P_curt"] = _v(model.P_curt[t])
        if kappa_exp is not None:
            row["allow_export"] = kappa_exp
        rows.append(row)

    return pd.DataFrame(rows)


def multi_model_to_dataframes(model: pyo.ConcreteModel) -> Dict[str, pd.DataFrame]:
    """
    Export a multi-house model into one dataframe per house.
    """
    if not hasattr(model, "H") or not hasattr(model, "T"):
        raise ValueError("Model does not look like a multi-house model (missing H/T)")

    kappa_exp = None
    if hasattr(model, "kappa_exp"):
        try:
            kappa_exp = int(round(_v(model.kappa_exp)))
        except Exception:
            kappa_exp = None

    has_p2p = hasattr(model, "c_p2p_buy") and hasattr(model, "c_p2p_sell") and hasattr(model, "c_p2p_fee")

    dfs: Dict[str, pd.DataFrame] = {}
    for h in list(model.H):
        rows = []
        for t in model.T:
            row = {
                "t": int(t),
                "Load": _v(model.Load[h, t]),
                "PV": _v(model.PV[h, t]),
                "c_grid": _v(model.c_grid[t]),
                "c_sell": _v(model.c_sell[t]),
                "P_imp": _v(model.P_imp[h, t]),
                "P_exp": _v(model.P_exp[h, t]),
                "P_ch": _v(model.P_ch[h, t]),
                "P_dis": _v(model.P_dis[h, t]),
                "E": _v(model.E[h, t]),
                "x": int(round(_v(model.x[h, t]))),
                "y": int(round(_v(model.y[h, t]))),
            }
            if hasattr(model, "P_curt"):
                row["P_curt"] = _v(model.P_curt[h, t])

            if hasattr(model, "P_share"):
                row["P_share"] = _v(model.P_share[h, t])
            if hasattr(model, "P_share_in"):
                row["P_share_in"] = _v(model.P_share_in[h, t])
            if hasattr(model, "P_share_out"):
                row["P_share_out"] = _v(model.P_share_out[h, t])

            if has_p2p:
                row["c_p2p_buy"] = _v(model.c_p2p_buy[t])
                row["c_p2p_sell"] = _v(model.c_p2p_sell[t])
                row["c_p2p_fee"] = _v(model.c_p2p_fee[t])

            if kappa_exp is not None:
                row["allow_export"] = kappa_exp

            rows.append(row)

        dfs[str(h)] = pd.DataFrame(rows)

    return dfs

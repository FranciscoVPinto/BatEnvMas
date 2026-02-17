from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo


def _v(x):
    try:
        return float(pyo.value(x))
    except Exception:
        return float(x)


def model_to_dataframe(model: pyo.ConcreteModel) -> pd.DataFrame:
    rows = []
    kappa_exp = None
    if hasattr(model, "kappa_exp"):
        try:
            kappa_exp = float(pyo.value(model.kappa_exp))
        except Exception:
            kappa_exp = None

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

        # Optional additions (newer model versions)
        if hasattr(model, "P_curt"):
            row["P_curt"] = _v(model.P_curt[t])
        if kappa_exp is not None:
            row["allow_export"] = int(round(kappa_exp))

        rows.append(row)

    return pd.DataFrame(rows)

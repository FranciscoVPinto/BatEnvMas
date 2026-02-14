from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo


def _v(x):
    # robust numeric extraction
    try:
        return float(pyo.value(x))
    except Exception:
        return float(x)


def model_to_dataframe(model: pyo.ConcreteModel) -> pd.DataFrame:
    rows = []
    for t in model.T:
        rows.append(
            {
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
        )
    return pd.DataFrame(rows)

from __future__ import annotations

from BatSimpleModel import SimpleBatteryModel
from bat_utils import solve_model, model_to_dataframe

def run_simple_battery(load, pv, c_grid, c_sell, params, solver="highs", tee=False):
    builder = SimpleBatteryModel(**params)
    m = builder.make_instance(load=load, pv=pv, c_grid=c_grid, c_sell=c_sell)
    results = solve_model(m, solver=solver, tee=tee)
    df = model_to_dataframe(m)
    return m, results, df


if __name__ == "__main__":
    import os
    import pandas as pd

    pv_df = pd.read_csv("Data/pv_gen.csv", header=None)
    load_df = pd.read_csv("Data/load_cons.csv", header=None)

    pv_series = pv_df.iloc[:500, 0]          # or: pv_df.sum(axis=1)
    load_series = load_df.iloc[:500, 0]      # or: load_df.sum(axis=1)

    T = len(load_series)

    c_grid = [0.25] * T   # buy price €/kWh (example)
    c_sell = [0.05] * T   # sell price €/kWh (example)

    params = dict(
        dt=1.0,
        P_ch_max=2.0,
        P_dis_max=2.0,
        E_min=0.0,
        E_max=5.0,
        eta_ch=0.95,
        eta_dis=0.95,
        E_init=2.5,
        P_grid_max=10.0,
    )

    m, results, df = run_simple_battery(
        load=load_series,
        pv=pv_series,
        c_grid=c_grid,
        c_sell=c_sell,
        params=params,
        solver="highs",
        tee=False
    )

    os.makedirs("plots", exist_ok=True)
    df.to_csv("plots/results.csv", index=False)

    from ploty import make_plot_simple
    make_plot_simple(df, outpath="plots/simple_battery.png")

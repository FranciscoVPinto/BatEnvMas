from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_plot_simple(df: pd.DataFrame, outpath: str | Path):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("t").reset_index(drop=True)
    x = df["t"].to_numpy()

    def get(col: str):
        return df[col].to_numpy() if col in df.columns else np.zeros(len(df))

    Load = get("Load")
    PV = get("PV")
    Pimp = get("P_imp")
    Pexp = get("P_exp")
    Pch = get("P_ch")
    Pdis = get("P_dis")
    E = get("E")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(x, Load, label="Load")
    ax1.plot(x, PV, label="PV")
    ax1.plot(x, Pimp, label="Grid import (P_imp)")
    ax1.plot(x, Pexp, label="Grid export (P_exp)")
    ax1.plot(x, Pch, label="Charge (P_ch)")
    ax1.plot(x, Pdis, label="Discharge (P_dis)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Power (kW)")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, E, label="Energy (E)")
    ax2.set_ylabel("Energy (kWh)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

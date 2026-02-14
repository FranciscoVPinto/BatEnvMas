# ploty.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_plot(df: pd.DataFrame, t0: int, t1: int, battery_id, include_export_in_pool: bool = True):
    """
    Função de plotting (mantida separada do modelo).

    include_export_in_pool=True -> PV_from_pool = SharedToLoad + posEInPV + negNetLoad
    include_export_in_pool=False -> PV_from_pool = SharedToLoad + posEInPV
    """
    hrs = np.arange(t0, t1)
    colors = sns.color_palette()
    idx = slice(t0, t1)

    # convenience accessor (graceful default to zeros if column missing)
    def col(name, default=0.0):
        if (battery_id, name) in df.columns:
            return df[(battery_id, name)]
        return pd.Series(default, index=df.index, dtype=float)

    load   = col('load')
    PVraw  = col('PV')
    posNL  = col('posNetLoad')
    negNL  = col('negNetLoad')
    soc    = col('SOC')
    tar    = col('tar')

    SharedToLoad = col('SharedToLoad')
    posEInPV     = col('posEInPV')

    if include_export_in_pool:
        PV_from_pool = SharedToLoad + posEInPV + negNL
        pool_label = 'PV from pool (incl. export)'
        pool_style = {'linestyle': '--'}
    else:
        PV_from_pool = SharedToLoad + posEInPV
        pool_label = 'PV from pool (no export)'
        pool_style = {'linestyle': '--'}

    action_terms = [col(n) for n in ['posEInGrid', 'posEInPV', 'negEOutLocal']]
    Action = sum(action_terms)

    load_s   = load[idx]
    PVraw_s  = PVraw[idx]
    PVpool_s = PV_from_pool[idx]
    posNL_s  = posNL[idx]
    negNL_s  = negNL[idx]
    soc_s    = soc[idx]
    tar_s    = tar[idx]
    act_s    = Action[idx]

    fig = plt.figure(figsize=(14, 18))
    fig.suptitle(f'Agent: {battery_id}', fontsize=16)

    ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=3)
    ax1.plot(hrs, load_s.values,            color=colors[0], label='Load')
    ax1.plot(hrs, PVraw_s.values,           color=colors[1], label='PV (raw)')
    ax1.plot(hrs, PVpool_s.values,          color=colors[6], label=pool_label, **pool_style)
    ax1.plot(hrs, posNL_s.values,           color=colors[2], label='PosNetLoad (import)')
    ax1.plot(hrs, negNL_s.values,           color=colors[3], label='NegNetLoad (export)')
    ax1.plot(hrs, act_s.values,             color=colors[5], label='BatteryAction')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('kWh')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(hrs, soc_s.values, color=colors[4], label='SOC')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('SOC (kWh)')

    ax3 = plt.subplot2grid((10, 1), (3, 0), rowspan=3)
    ax3.plot(hrs, soc_s.values, color=colors[4], label='SOC')
    ax3.plot(hrs, act_s.values, color=colors[5], label='BatteryAction')
    ax3.legend(loc='upper left')
    ax3.set_ylabel('kWh')
    ax3.grid(True)

    ax4 = ax3.twinx()
    ax4.plot(hrs, tar_s.values, color=colors[3], linestyle='--', label='Buy Price')
    ax4.legend(loc='upper right')
    ax4.set_ylabel('Price')
    ax4.grid(axis='x', linewidth=1.5, color='black', linestyle='dashed')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def make_plot_simple(df: pd.DataFrame, outpath: str | None = None):
    """
    Plot simples para o DataFrame do SimpleBatteryModel (bat_utils.model_to_dataframe).
    Espera colunas: t, Load, PV, P_imp, P_exp, P_ch, P_dis, E, c_grid, c_sell, ...
    """
    # garantir ordenação por tempo
    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)
        x = df["t"].to_numpy()
    else:
        x = np.arange(len(df))

    def get(col):
        return df[col].to_numpy() if col in df.columns else np.zeros(len(df))

    Load = get("Load")
    PV   = get("PV")
    Pimp = get("P_imp")
    Pexp = get("P_exp")
    Pch  = get("P_ch")
    Pdis = get("P_dis")
    E    = get("E")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(x, Load, label="Load")
    ax1.plot(x, PV,   label="PV")
    ax1.plot(x, Pimp, label="Grid import (P_imp)")
    ax1.plot(x, Pexp, label="Grid export (P_exp)")
    ax1.plot(x, Pch,  label="Charge (P_ch)")
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

    if outpath:
        fig.savefig(outpath, dpi=200)
    else:
        plt.show()

    plt.close(fig)

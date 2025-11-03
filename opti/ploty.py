import pyomo.environ as en
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from battery import Battery

def make_plot(df, t0, t1, battery_id, include_export_in_pool=True):
    """
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
        # default series with same index as df
        return pd.Series(default, index=df.index, dtype=float)

    # core series
    load   = col('load')
    PVraw  = col('PV')                  # agent's own generation (raw)
    posNL  = col('posNetLoad')          # grid imports
    negNL  = col('negNetLoad')          # export credited to this agent (from pool split)
    soc    = col('SOC')
    tar    = col('tar')

    # pool allocation pieces (may not exist if not exported from model; we default to 0)
    SharedToLoad = col('SharedToLoad')
    posEInPV     = col('posEInPV')

    # PV coming from community pool (excedent)
    if include_export_in_pool:
        PV_from_pool = SharedToLoad + posEInPV + negNL
        pool_label = 'PV from pool (incl. export)'
        pool_style = {'linestyle': '--'}
    else:
        PV_from_pool = SharedToLoad + posEInPV
        pool_label = 'PV from pool (no export)'
        pool_style = {'linestyle': '--'}

    # battery action proxy (charge positive, discharge negative if negEOutLocal <= 0)
    # sum of inflows to batt (posEInGrid + posEInPV) plus outflow (negEOutLocal, typically ≤ 0)
    action_terms = []
    for name in ['posEInGrid', 'posEInPV', 'negEOutLocal']:
        s = col(name)
        action_terms.append(s)
    Action = sum(action_terms)

    # slice once for plotting
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

    # Top plot
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

    # SOC on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(hrs, soc_s.values, color=colors[4], label='SOC')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('SOC (kWh)')

    # Middle plot (SOC & battery action)
    ax3 = plt.subplot2grid((10, 1), (3, 0), rowspan=3)
    ax3.plot(hrs, soc_s.values, color=colors[4], label='SOC')
    ax3.plot(hrs, act_s.values, color=colors[5], label='BatteryAction')
    ax3.legend(loc='upper left')
    ax3.set_ylabel('kWh')
    ax3.grid(True)

    # Tariff on secondary y-axis
    ax4 = ax3.twinx()
    ax4.plot(hrs, tar_s.values, color=colors[3], linestyle='--', label='Buy Price')
    ax4.legend(loc='upper right')
    ax4.set_ylabel('Price')
    ax4.grid(axis='x', linewidth=1.5, color='black', linestyle='dashed')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

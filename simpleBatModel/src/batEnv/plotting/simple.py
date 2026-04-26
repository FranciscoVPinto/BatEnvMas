from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _ensure_derived(df: pd.DataFrame, dt_hours: float) -> pd.DataFrame:
    out = df.copy()

    num_cols = ["P_imp", "P_exp", "P_ch", "P_dis", "P_curt", "Load", "PV", "E", "c_grid", "c_sell"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    if "P_imp" in out.columns and "P_exp" in out.columns:
        out["P_net_grid"] = out["P_imp"] - out["P_exp"]
        out["P_simul_imp_exp"] = np.minimum(out["P_imp"].to_numpy(), out["P_exp"].to_numpy())

    if "c_grid" in out.columns and "P_imp" in out.columns:
        c_sell = out["c_sell"] if "c_sell" in out.columns else 0.0
        P_exp = out["P_exp"] if "P_exp" in out.columns else 0.0
        out["cost_step"] = (out["c_grid"] * out["P_imp"] - c_sell * P_exp) * float(dt_hours)
        out["cost_cum"] = out["cost_step"].cumsum()

    return out


def compute_summary_metrics(df: pd.DataFrame, dt_hours: float) -> Dict[str, Any]:
    d = _ensure_derived(df, dt_hours=dt_hours)

    def _E(col: str) -> float:
        if col not in d.columns:
            return 0.0
        return float(d[col].sum() * float(dt_hours))

    out: Dict[str, Any] = {
        "E_load_kWh": _E("Load"),
        "E_pv_kWh": _E("PV"),
        "E_imp_kWh": _E("P_imp"),
        "E_exp_kWh": _E("P_exp"),
        "E_ch_kWh": _E("P_ch"),
        "E_dis_kWh": _E("P_dis"),
        "E_curt_kWh": _E("P_curt"),
    }

    if out["E_pv_kWh"] > 0:
        out["Curt_frac_of_PV"] = float(out["E_curt_kWh"] / out["E_pv_kWh"])

    if "E" in d.columns and len(d["E"]) > 0:
        out["E_end_kWh"] = float(d["E"].iloc[-1])
        out["E_min_kWh"] = float(d["E"].min())
        out["E_max_kWh"] = float(d["E"].max())

    if "cost_step" in d.columns and len(d["cost_step"]) > 0:
        out["Cost_total_EUR"] = float(d["cost_step"].sum())
        out["Cost_min_step_EUR"] = float(d["cost_step"].min())
        out["Cost_max_step_EUR"] = float(d["cost_step"].max())

    if "P_net_grid" in d.columns and len(d["P_net_grid"]) > 0:
        out["P_net_grid_max_kW"] = float(d["P_net_grid"].max())
        out["P_net_grid_min_kW"] = float(d["P_net_grid"].min())

    if "P_imp" in d.columns and len(d["P_imp"]) > 0:
        out["P_imp_max_kW"] = float(d["P_imp"].max())
    if "P_exp" in d.columns and len(d["P_exp"]) > 0:
        out["P_exp_max_kW"] = float(d["P_exp"].max())

    if "P_simul_imp_exp" in d.columns:
        out["E_simul_imp_exp_kWh"] = float(d["P_simul_imp_exp"].sum() * float(dt_hours))

    return out

def _legend_both(ax, ax2, *, loc="upper left"):
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc=loc)


def plot_house_per_case(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "",
    dt_hours: float = 1.0,
    E_min: Optional[float] = None,
    E_max: Optional[float] = None,
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    d = _ensure_derived(df, dt_hours=dt_hours)

    # time axis (days)
    if "t" in d.columns:
        t = pd.to_numeric(d["t"], errors="coerce").fillna(0).to_numpy()
        x_hours = (t - t.min()) * float(dt_hours)
    else:
        x_hours = np.arange(len(d)) * float(dt_hours)

    x = x_hours / 24.0  # days

    # helper arrays
    Load = d["Load"].to_numpy() if "Load" in d.columns else None
    PV = d["PV"].to_numpy() if "PV" in d.columns else None
    P_curt = d["P_curt"].to_numpy() if "P_curt" in d.columns else None
    P_imp = d["P_imp"].to_numpy() if "P_imp" in d.columns else None
    P_exp = d["P_exp"].to_numpy() if "P_exp" in d.columns else None
    P_ch = d["P_ch"].to_numpy() if "P_ch" in d.columns else None
    P_dis = d["P_dis"].to_numpy() if "P_dis" in d.columns else None

    fig = plt.figure(figsize=(13, 10))

    # 1) Load / PV / Curtailment
    ax1 = fig.add_subplot(4, 1, 1)
    if Load is not None:
        ax1.plot(x, Load, label="Load (kW)")
    if PV is not None:
        ax1.plot(x, PV, label="PV allocated (kW)")
    if P_curt is not None:
        ax1.plot(x, P_curt, label="PV curtailed (kW)")
    ax1.set_ylabel("kW")
    ax1.set_title(title or "House")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Grid (net with sign)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    if P_imp is not None and P_exp is not None:
        P_net = P_imp - P_exp
        ax2.plot(x, P_net, label="Grid net (+import / −export) (kW)")
        ax2.axhline(0, linewidth=1)
    else:
        if P_imp is not None:
            ax2.plot(x, P_imp, label="Import (kW)")
        if P_exp is not None:
            ax2.plot(x, -P_exp, label="−Export (kW)")
            ax2.axhline(0, linewidth=1)
    ax2.set_ylabel("kW")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) Battery power + SOC (or E)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    if P_ch is not None or P_dis is not None:
        # show one signed curve: +discharge, -charge
        net_bat = (P_dis if P_dis is not None else 0.0) - (P_ch if P_ch is not None else 0.0)
        ax3.plot(x, net_bat, label="Battery net (+dis / −ch) (kW)")
        ax3.axhline(0, linewidth=1)
    ax3.set_ylabel("kW")
    ax3.grid(True, alpha=0.3)

    ax3b = ax3.twinx()
    if "E" in d.columns:
        E = pd.to_numeric(d["E"], errors="coerce").fillna(0.0).to_numpy()
        if (E_min is not None) and (E_max is not None) and (E_max > E_min):
            soc = (E - float(E_min)) / (float(E_max) - float(E_min))
            ax3b.plot(x, 100.0 * soc, label="SOC (%)")
            ax3b.set_ylabel("%")
            ax3b.set_ylim(-5, 105)
        else:
            ax3b.plot(x, E, label="Energy E (kWh)")
            ax3b.set_ylabel("kWh")

        # show bounds if provided
        if E_min is not None:
            ax3b.axhline(100.0 * 0 if (E_min is not None and E_max is not None and E_max > E_min) else float(E_min),
                         linewidth=1, linestyle="--", label="E_min")
        if E_max is not None:
            ax3b.axhline(100.0 if (E_min is not None and E_max is not None and E_max > E_min) else float(E_max),
                         linewidth=1, linestyle="--", label="E_max")

    _legend_both(ax3, ax3b, loc="upper left")

    # 4) Tariffs + cost (with legends!)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
    if "c_grid" in d.columns:
        ax4.plot(x, d["c_grid"].to_numpy(), label="c_grid (€/kWh)")
    if "c_sell" in d.columns:
        ax4.plot(x, d["c_sell"].to_numpy(), label="c_sell (€/kWh)")
    ax4.set_ylabel("€/kWh")
    ax4.grid(True, alpha=0.3)

    ax4b = ax4.twinx()
    if "cost_cum" in d.columns:
        ax4b.plot(x, d["cost_cum"].to_numpy(), label="Cost cumulative (€)")
    ax4b.set_ylabel("€")

    _legend_both(ax4, ax4b, loc="upper left")
    ax4.set_xlabel("time (days)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
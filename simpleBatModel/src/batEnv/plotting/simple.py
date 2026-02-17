from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_derived(df: pd.DataFrame, dt_hours: float) -> pd.DataFrame:
    """
    Adds time-dependent derived columns used in plots/comparisons:
      - P_net_grid = P_imp - P_exp
      - P_simul_imp_exp = min(P_imp, P_exp)
      - cost_step = (c_grid*P_imp - c_sell*P_exp)*dt_hours
      - cost_cum = cumsum(cost_step)
    """
    out = df.copy()

    for c in ["P_imp", "P_exp", "P_ch", "P_dis", "P_curt", "Load", "PV", "E", "c_grid", "c_sell"]:
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
    """
    Summary KPIs (still useful), but computed from time-series (dt-aware).
    """
    d = _ensure_derived(df, dt_hours=dt_hours)

    def _E(col: str) -> float:
        if col not in d.columns:
            return 0.0
        return float(d[col].sum() * float(dt_hours))

    out: Dict[str, Any] = {}
    out["E_load_kWh"] = _E("Load")
    out["E_pv_kWh"] = _E("PV")
    out["E_imp_kWh"] = _E("P_imp")
    out["E_exp_kWh"] = _E("P_exp")
    out["E_ch_kWh"] = _E("P_ch")
    out["E_dis_kWh"] = _E("P_dis")
    out["E_curt_kWh"] = _E("P_curt")

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


def plot_house_per_case(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "",
    dt_hours: float = 1.0,
) -> None:
    """
    Single PNG per house (but with multiple stacked panels) for detailed time-dependent inspection.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    d = _ensure_derived(df, dt_hours=dt_hours)

    if "t" in d.columns:
        t = pd.to_numeric(d["t"], errors="coerce").fillna(0).to_numpy()
        x = (t - t.min()) * float(dt_hours)
    else:
        x = np.arange(len(d)) * float(dt_hours)

    fig = plt.figure(figsize=(13, 10))

    # 1) Load & PV
    ax1 = fig.add_subplot(4, 1, 1)
    if "Load" in d.columns:
        ax1.plot(x, d["Load"].to_numpy(), label="Load (kW)")
    if "PV" in d.columns:
        ax1.plot(x, d["PV"].to_numpy(), label="PV (kW)")
    if "P_curt" in d.columns:
        ax1.plot(x, d["P_curt"].to_numpy(), label="PV curtailed P_curt (kW)")
    ax1.set_ylabel("kW")
    ax1.set_title(title or "House")
    ax1.legend()

    # 2) Grid flows + net grid
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    if "P_imp" in d.columns:
        ax2.plot(x, d["P_imp"].to_numpy(), label="Import (kW)")
    if "P_exp" in d.columns:
        ax2.plot(x, d["P_exp"].to_numpy(), label="Export (kW)")
    if "P_net_grid" in d.columns:
        ax2.plot(x, d["P_net_grid"].to_numpy(), label="Net grid (Imp-Exp) (kW)")
    ax2.set_ylabel("kW")
    ax2.legend()

    # 3) Battery power + Energy
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    if "P_ch" in d.columns:
        ax3.plot(x, d["P_ch"].to_numpy(), label="Charge P_ch (kW)")
    if "P_dis" in d.columns:
        ax3.plot(x, d["P_dis"].to_numpy(), label="Discharge P_dis (kW)")
    ax3.set_ylabel("kW")
    ax3.legend(loc="upper left")

    ax3b = ax3.twinx()
    if "E" in d.columns:
        ax3b.plot(x, d["E"].to_numpy(), label="Energy E (kWh)")
        ax3b.set_ylabel("kWh")

    # 4) Prices + costs (step & cumulative)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
    if "c_grid" in d.columns:
        ax4.plot(x, d["c_grid"].to_numpy(), label="c_grid (€/kWh)")
    if "c_sell" in d.columns:
        ax4.plot(x, d["c_sell"].to_numpy(), label="c_sell (€/kWh)")
    ax4.set_ylabel("€/kWh")
    ax4.legend(loc="upper left")

    ax4b = ax4.twinx()
    if "cost_step" in d.columns:
        ax4b.plot(x, d["cost_step"].to_numpy(), label=f"cost_step (€) [dt={dt_hours}h]")
    if "cost_cum" in d.columns:
        ax4b.plot(x, d["cost_cum"].to_numpy(), label="cost_cum (€)")
    ax4b.set_ylabel("€")

    ax4.set_xlabel("time (hours)")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        sc = (out["E_pv_kWh"] - out["E_curt_kWh"] - out["E_exp_kWh"]) / out["E_pv_kWh"]
        out["Self_Consumption"] = float(max(0.0, min(1.0, sc)))

    # Aproveitamento do excedente PV pela bateria:
    # excedente = energia PV que nao foi consumida directamente (E_ch + E_exp + E_curt)
    # fracao_aproveitada = E_ch / excedente  (sem grid charging, E_ch = P_pv_to_bat)
    e_surplus = out["E_ch_kWh"] + out["E_exp_kWh"] + out["E_curt_kWh"]
    if e_surplus > 0:
        out["PV_surplus_kWh"] = float(e_surplus)
        out["Surplus_captured_frac"] = float(
            max(0.0, min(1.0, out["E_ch_kWh"] / e_surplus))
        )

    if out["E_load_kWh"] > 0:
        ss = (out["E_load_kWh"] - out["E_imp_kWh"]) / out["E_load_kWh"]
        out["Self_Sufficiency"] = float(max(0.0, min(1.0, ss)))

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


# SoC bin colour palette — low (red), mid (green), high (orange)
_BIN_COLOURS = ["#ffcccc", "#ccffcc", "#ffe0b2", "#cce0ff", "#f5ccff"]


def _draw_soc_bin_zones(
    ax: plt.Axes,
    soc_breakpoints: List[float],
    *,
    alpha: float = 0.18,
) -> None:
    """
    Draw horizontal shaded bands on a SOC (%) axis for each PWL bin.

    soc_breakpoints: fractions in [0, 1], e.g. [0.0, 0.2, 0.8, 1.0]
    The axis is assumed to be in % (0–100).
    """
    bkpts_pct = [100.0 * b for b in soc_breakpoints]
    n_bins = len(bkpts_pct) - 1
    for k in range(n_bins):
        colour = _BIN_COLOURS[k % len(_BIN_COLOURS)]
        ax.axhspan(bkpts_pct[k], bkpts_pct[k + 1], facecolor=colour, alpha=alpha, zorder=0)
    # Draw bin boundary lines (excluding 0 % and 100 %)
    for bkpt in bkpts_pct[1:-1]:
        ax.axhline(bkpt, color="#888888", linewidth=0.8, linestyle="--", zorder=1)


def plot_house_per_case(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "",
    dt_hours: float = 1.0,
    E_min: Optional[float] = None,
    E_max: Optional[float] = None,
    pwl_soc_breakpoints: Optional[List[float]] = None,
) -> None:
    """
    2-panel per-house timeseries:
      Painel 1 — Potências: Load, PV, importação/exportação rede, carga/descarga bateria
      Painel 2 — Estado de carga (SoC %) + custo acumulado (€)
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    d = _ensure_derived(df, dt_hours=dt_hours)

    if "t" in d.columns:
        t = pd.to_numeric(d["t"], errors="coerce").fillna(0).to_numpy()
        x = (t - t.min()) * float(dt_hours) / 24.0   # dias
    else:
        x = np.arange(len(d)) * float(dt_hours) / 24.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # --- Painel 1: Potências ---
    if "Load" in d.columns:
        ax1.plot(x, d["Load"].to_numpy(), label="Carga (kW)", color="#333333")
    if "PV" in d.columns:
        ax1.plot(x, d["PV"].to_numpy(), label="PV (kW)", color="#f4a11d", linewidth=1.2)
    if "P_imp" in d.columns:
        ax1.plot(x, d["P_imp"].to_numpy(), label="Importação (kW)", color="#1f77b4", alpha=0.8)
    if "P_exp" in d.columns:
        ax1.plot(x, -d["P_exp"].to_numpy(), label="−Exportação (kW)", color="#17becf", alpha=0.8)
    if "P_ch" in d.columns and "P_dis" in d.columns:
        net_bat = d["P_dis"].to_numpy() - d["P_ch"].to_numpy()
        ax1.plot(x, net_bat, label="Bateria (+dis/−ch) (kW)", color="#9467bd", linewidth=1.0, alpha=0.8)
    ax1.axhline(0, linewidth=0.8, color="#aaaaaa")
    ax1.set_ylabel("kW")
    ax1.set_title(title or "Fração")
    ax1.grid(True, alpha=0.25)
    ax1.legend(ncol=3, fontsize=8, loc="upper right")

    # --- Painel 2: SoC + custo acumulado ---
    ax2b = ax2.twinx()
    if "E" in d.columns:
        E = pd.to_numeric(d["E"], errors="coerce").fillna(0.0).to_numpy()
        if E_min is not None and E_max is not None and E_max > E_min:
            soc = 100.0 * (E - float(E_min)) / (float(E_max) - float(E_min))
            if pwl_soc_breakpoints and len(pwl_soc_breakpoints) >= 2:
                _draw_soc_bin_zones(ax2, pwl_soc_breakpoints)
            ax2.plot(x, soc, label="SoC (%)", color="#ff7f0e", linewidth=1.5, zorder=3)
            ax2.set_ylabel("SoC (%)")
            ax2.set_ylim(-5, 105)
        else:
            ax2.plot(x, E, label="E (kWh)", color="#ff7f0e", linewidth=1.5)
            ax2.set_ylabel("kWh")
    if "cost_cum" in d.columns:
        ax2b.plot(x, d["cost_cum"].to_numpy(), label="Custo acum. (€)", color="#2ca02c", linestyle="--")
        ax2b.set_ylabel("€")
    _legend_both(ax2, ax2b, loc="upper left")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlabel("tempo (dias)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_derived_columns(df: pd.DataFrame, dt_hours: float = 1.0) -> pd.DataFrame:
    """
    Adds derived columns using dt_hours so costs are in EUR and energy in kWh scaling is consistent.
    """
    df = df.copy()

    if {"c_grid", "P_imp", "c_sell", "P_exp"}.issubset(df.columns):
        # €/kWh * kW * h = €
        df["cost_step"] = (df["c_grid"] * df["P_imp"] - df["c_sell"] * df["P_exp"]) * dt_hours
        df["cost_cum"] = df["cost_step"].cumsum()

    if {"P_imp", "P_exp"}.issubset(df.columns):
        df["P_net_grid"] = df["P_imp"] - df["P_exp"]

    return df


def compute_summary_metrics(df: pd.DataFrame, dt_hours: float = 1.0) -> dict:
    df = add_derived_columns(df, dt_hours=dt_hours)
    out = {}

    if "P_imp" in df.columns:
        out["E_imp_kWh"] = float(df["P_imp"].sum() * dt_hours)
    if "P_exp" in df.columns:
        out["E_exp_kWh"] = float(df["P_exp"].sum() * dt_hours)
    if "P_ch" in df.columns:
        out["E_ch_kWh"] = float(df["P_ch"].sum() * dt_hours)
    if "P_dis" in df.columns:
        out["E_dis_kWh"] = float(df["P_dis"].sum() * dt_hours)

    if "cost_step" in df.columns:
        out["Cost_total_EUR"] = float(df["cost_step"].sum())

    if "E" in df.columns:
        out["E_min_kWh"] = float(df["E"].min())
        out["E_max_kWh"] = float(df["E"].max())
        out["E_end_kWh"] = float(df["E"].iloc[-1])

    return out


def plot_house_per_case(df: pd.DataFrame, outpath: str | Path, title: str | None = None, dt_hours: float = 1.0):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = add_derived_columns(df, dt_hours=dt_hours)
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
    if title:
        ax1.set_title(title)

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


def plot_compare_timeseries(
    dfs: Dict[str, pd.DataFrame],
    variable: str,
    outpath: str | Path,
    title: str | None = None,
    dt_hours_by_case: Optional[Dict[str, float]] = None,
):
    """
    Compare a variable across multiple cases for the same house.
    If variable is derived (cost_step/cost_cum), dt_hours_by_case is used per case.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    if title:
        ax.set_title(title)

    for case_name, df in dfs.items():
        dt_h = 1.0
        if dt_hours_by_case and case_name in dt_hours_by_case:
            dt_h = float(dt_hours_by_case[case_name])

        df2 = add_derived_columns(df, dt_hours=dt_h)
        df2 = df2.sort_values("t").reset_index(drop=True)

        if variable not in df2.columns:
            continue

        ax.plot(df2["t"].to_numpy(), df2[variable].to_numpy(), label=case_name)

    ax.set_xlabel("t")
    ax.set_ylabel(variable)
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_compare_metrics(metrics: pd.DataFrame, metric: str, outpath: str | Path, title: str | None = None):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if metric not in metrics.columns:
        raise ValueError(f"Metric '{metric}' not in metrics dataframe columns")

    fig, ax = plt.subplots(figsize=(14, 6))
    if title:
        ax.set_title(title)

    x = np.arange(len(metrics.index))
    y = metrics[metric].to_numpy()

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.index, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y")

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

from .simple import _ensure_derived


def plot_compare_timeseries(
    dfs_by_case: Dict[str, pd.DataFrame],
    *,
    variable: str,
    outpath: str | Path,
    title: str = "",
    dt_hours_by_case: Dict[str, float] | None = None,
) -> None:
    """
    Overlay time series across cases for a given variable.
    Supports derived variables: cost_step, cost_cum, P_net_grid, P_simul_imp_exp.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if dt_hours_by_case is None:
        dt_hours_by_case = {}

    # align to min length
    min_len = min(len(df) for df in dfs_by_case.values())
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    for case_name, df in dfs_by_case.items():
        dt = float(dt_hours_by_case.get(case_name, 1.0))
        d = _ensure_derived(df.iloc[:min_len].copy(), dt_hours=dt)

        if "t" in d.columns:
            t = pd.to_numeric(d["t"], errors="coerce").fillna(0).to_numpy()
            x = (t - t.min()) * dt
        else:
            x = list(range(min_len))

        if variable not in d.columns:
            continue

        ax.plot(x, pd.to_numeric(d[variable], errors="coerce").fillna(0.0).to_numpy(), label=case_name)

    ax.set_title(title or f"Compare {variable}")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel(variable)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_compare_metrics(
    metrics_df: pd.DataFrame,
    *,
    metric: str,
    outpath: str | Path,
    title: str = "",
) -> None:
    """
    Bar chart comparing a metric across cases.
    Expects metrics_df index = case, column = metric.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if metric not in metrics_df.columns:
        return

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    x = list(metrics_df.index.astype(str))
    y = pd.to_numeric(metrics_df[metric], errors="coerce").fillna(0.0).to_list()

    ax.bar(x, y)
    ax.set_title(title or metric)
    ax.set_xlabel("case")
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

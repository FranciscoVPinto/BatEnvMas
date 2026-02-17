from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

from .simple import _ensure_derived


def _wrap_case_label(case_name: str, group: int = 2, max_lines: int = 4) -> str:
    """
    Make long case names readable by splitting on '_' and joining in multi-lines.

    Example:
      pv07_dyn_instant_no_export
      -> pv07_dyn
         instant_no
         export

    group=2 groups tokens in pairs to keep lines not too short.
    max_lines limits vertical growth (keeps plots sane).
    """
    parts = str(case_name).split("_")
    if len(parts) <= group:
        return case_name

    lines = ["_".join(parts[i : i + group]) for i in range(0, len(parts), group)]

    # Keep at most max_lines; if too many, merge tail
    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = "_".join(lines[max_lines - 1 :])
        lines = head + [tail]

    return "\n".join(lines)


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

        ax.plot(
            x,
            pd.to_numeric(d[variable], errors="coerce").fillna(0.0).to_numpy(),
            label=case_name,
        )

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

    Improvements for readability:
      - Wrap long case labels into multiple lines
      - Increase figure size dynamically based on number of cases
      - Switch to horizontal bars if there are many cases
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if metric not in metrics_df.columns:
        return

    raw_labels = list(metrics_df.index.astype(str))
    y = pd.to_numeric(metrics_df[metric], errors="coerce").fillna(0.0).to_list()

    n = len(raw_labels)

    # If many cases, horizontal is always readable.
    use_horizontal = n >= 12

    if use_horizontal:
        # Height grows with number of cases
        fig_h = max(5.0, 0.45 * n)
        fig = plt.figure(figsize=(12, fig_h))
        ax = fig.add_subplot(111)

        labels = [_wrap_case_label(s, group=2, max_lines=5) for s in raw_labels]
        ax.barh(labels, y)

        ax.set_title(title or metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("case")
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)
        return

    # Vertical bars: widen the figure a bit and wrap labels
    fig_w = max(12.0, 1.6 * n)
    fig = plt.figure(figsize=(fig_w, 6.0))
    ax = fig.add_subplot(111)

    labels = [_wrap_case_label(s, group=2, max_lines=4) for s in raw_labels]
    x = list(range(n))

    ax.bar(x, y)
    ax.set_title(title or metric)
    ax.set_xlabel("case")
    ax.set_ylabel(metric)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    # Give extra room at the bottom for multi-line labels
    fig.subplots_adjust(bottom=0.28)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

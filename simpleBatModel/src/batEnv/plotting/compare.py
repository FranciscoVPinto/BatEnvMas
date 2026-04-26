from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .simple import _ensure_derived


_COMMUNITY_ID = "_COMMUNITY"


def _wrap_case_label(case_name: str, group: int = 2, max_lines: int = 4) -> str:
    parts = str(case_name).split("_")
    if len(parts) <= group:
        return str(case_name)

    lines = ["_".join(parts[i : i + group]) for i in range(0, len(parts), group)]
    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = "_".join(lines[max_lines - 1 :])
        lines = head + [tail]
    return "\n".join(lines)


def _short_case(name: str) -> str:
    """Strip the `b8_` prefix from cases for compactness in summary plots."""
    s = str(name)
    return s[3:] if s.startswith("b8_") else s


# ---------- timeseries comparisons across cases ----------

def plot_compare_timeseries(
    dfs_by_case: Dict[str, pd.DataFrame],
    *,
    variable: str,
    outpath: str | Path,
    title: str = "",
    dt_hours_by_case: Dict[str, float] | None = None,
) -> None:
    """
    Compare a single variable across cases.

    When the legend includes `_COMMUNITY`, that series is drawn with a thicker
    black dashed line so it stands out from the per-house curves.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if dt_hours_by_case is None:
        dt_hours_by_case = {}

    if not dfs_by_case:
        return

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

        y = pd.to_numeric(d[variable], errors="coerce").fillna(0.0).to_numpy()
        # Distinctive style for the community aggregate: black, thicker, dashed.
        if str(case_name) == _COMMUNITY_ID:
            ax.plot(x, y, label=case_name, color="black", linewidth=2.0, linestyle="--", zorder=5)
        else:
            ax.plot(x, y, label=case_name, linewidth=1.0, alpha=0.85)

    ax.set_title(title or f"Compare {variable}")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel(variable)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ---------- metric bars across cases ----------

def plot_compare_metrics(
    metrics_df: pd.DataFrame,
    *,
    metric: str,
    outpath: str | Path,
    title: str = "",
    sort_by_value: bool = False,
    higher_is_better: Optional[bool] = None,
) -> None:
    """
    Bar chart of a single metric across cases.

    Parameters
    ----------
    sort_by_value :
        When True, cases are sorted by metric value (ascending). Reading the
        ranking becomes immediate.
    higher_is_better :
        Only meaningful when `sort_by_value=True`. When provided, the "best"
        bar is coloured green and the "worst" red — useful for quick triage
        in figures shown to non-experts.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if metric not in metrics_df.columns:
        return

    df = metrics_df.copy()
    if sort_by_value:
        df = df.sort_values(metric, ascending=True)

    raw_labels = list(df.index.astype(str))
    y = pd.to_numeric(df[metric], errors="coerce").fillna(0.0).to_numpy()
    n = len(raw_labels)
    if n == 0:
        return

    use_horizontal = n >= 12

    if use_horizontal:
        fig_h = max(5.0, 0.45 * n)
        fig = plt.figure(figsize=(12, fig_h))
        ax = fig.add_subplot(111)

        labels = [_wrap_case_label(s, group=2, max_lines=6) for s in raw_labels]
        bars = ax.barh(labels, y)
        ax.set_title(title or metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("case")
    else:
        fig_w = max(12.0, 1.6 * n)
        fig = plt.figure(figsize=(fig_w, 6.0))
        ax = fig.add_subplot(111)

        labels = [_wrap_case_label(s, group=2, max_lines=4) for s in raw_labels]
        x = list(range(n))
        bars = ax.bar(x, y)
        ax.set_title(title or metric)
        ax.set_xlabel("case")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        fig.subplots_adjust(bottom=0.28)

    # Optional best/worst highlight
    if sort_by_value and higher_is_better is not None and n >= 2:
        best = int(np.argmax(y) if higher_is_better else np.argmin(y))
        worst = int(np.argmin(y) if higher_is_better else np.argmax(y))
        bars[best].set_color("#2ca02c")
        bars[worst].set_color("#d62728")

    ax.grid(True, alpha=0.3, axis="x" if use_horizontal else "y")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ---------- per-case alpha allocation (PV sharing) ----------

def plot_alpha_allocation(
    house_dfs: Dict[str, pd.DataFrame],
    outpath: str | Path,
    *,
    title: str = "",
    dt_hours: float = 1.0,
    exclude_keys: tuple = (_COMMUNITY_ID,),
) -> None:
    """
    Stacked-area chart of alpha_i(t) = PV_i(t) / sum_j PV_j(t) across the houses
    of a single case. Visualises HOW the sharing strategy actually allocated PV
    over time — flat bands for fixed-alpha schemes, oscillating bands for
    consumption-based fallbacks.

    Houses listed in `exclude_keys` (e.g. the community aggregate) are skipped.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    real_houses = [(h, df) for h, df in house_dfs.items() if h not in exclude_keys]
    if not real_houses:
        return

    # Align by length and t axis
    min_len = min(len(df) for _, df in real_houses)

    pv_matrix = []
    house_labels = []
    t_axis: Optional[np.ndarray] = None
    for h, df in real_houses:
        d = df.iloc[:min_len]
        if "PV" not in d.columns:
            continue
        pv_matrix.append(pd.to_numeric(d["PV"], errors="coerce").fillna(0.0).to_numpy())
        house_labels.append(h)
        if t_axis is None and "t" in d.columns:
            t = pd.to_numeric(d["t"], errors="coerce").fillna(0).to_numpy()
            t_axis = (t - t.min()) * float(dt_hours)

    if not pv_matrix:
        return

    pv_arr = np.vstack(pv_matrix)  # shape (n_houses, T)
    pv_total = pv_arr.sum(axis=0)

    # alpha_i(t): when total is zero, fall back to equal split so the plot
    # remains continuous and clearly readable.
    n_h = pv_arr.shape[0]
    eq = 1.0 / n_h
    alpha = np.where(pv_total > 0, pv_arr / np.where(pv_total > 0, pv_total, 1.0), eq)

    if t_axis is None:
        t_axis = np.arange(pv_arr.shape[1]) * float(dt_hours)
    x_days = t_axis / 24.0

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

    ax_top.stackplot(x_days, alpha, labels=house_labels, alpha=0.85)
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("alpha_i(t) (fraction of PV total)")
    ax_top.set_title(title or "PV allocation alpha(t) across houses")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right", ncol=min(4, n_h), fontsize=8)

    # Bottom: PV total kW for context — explains when alpha matters.
    ax_bot.plot(x_days, pv_total, color="#888888", linewidth=1.0)
    ax_bot.fill_between(x_days, 0, pv_total, color="#cccccc", alpha=0.5)
    ax_bot.set_ylabel("PV total (kW)")
    ax_bot.set_xlabel("time (days)")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------- one-page summary dashboard ----------

def plot_summary_dashboard(
    metrics_df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "",
    sort_by: str = "Cost_total_EUR",
) -> None:
    """
    One-figure summary across all cases for the *community* aggregate.

    Reads the multi-index metrics DataFrame produced by render_results
    (level 'house' contains '_COMMUNITY' rows) and draws a 2x2 grid:
       [Self_Sufficiency_COMM]  [Self_Consumption_COMM]
       [Cost_total_EUR (COMM)]  [Cost_total_EUR_CV_house]

    Cases are sorted by `sort_by` (ascending) when that column is available
    so the cheapest strategy appears first.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if metrics_df.empty:
        return

    try:
        comm = metrics_df.xs(_COMMUNITY_ID, level="house").copy()
    except KeyError:
        return

    if sort_by in comm.columns:
        comm = comm.sort_values(sort_by, ascending=True)

    cases = [_short_case(c) for c in comm.index.astype(str)]
    n = len(cases)

    panels = [
        ("Self_Sufficiency_COMM", "Self-sufficiency (community)", "fraction", (0, 1)),
        ("Self_Consumption_COMM", "Self-consumption (community)", "fraction", (0, 1)),
        ("Cost_total_EUR",        "Total community cost",         "EUR",      None),
        ("Cost_total_EUR_CV_house", "Cost dispersion across houses (CV)", "CV", None),
    ]

    fig_w = max(11.0, 0.9 * n + 6.0)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 8.5))
    axes = axes.flatten()

    for ax, (col, subtitle, ylabel, ylim) in zip(axes, panels):
        if col not in comm.columns:
            ax.text(0.5, 0.5, f"{col}\nnot available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(subtitle)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        y = pd.to_numeric(comm[col], errors="coerce").fillna(0.0).to_numpy()
        bars = ax.bar(range(n), y)

        if n >= 2:
            higher_is_better = col in ("Self_Sufficiency_COMM", "Self_Consumption_COMM")
            best = int(np.argmax(y) if higher_is_better else np.argmin(y))
            worst = int(np.argmin(y) if higher_is_better else np.argmax(y))
            bars[best].set_color("#2ca02c")
            bars[worst].set_color("#d62728")

        ax.set_title(subtitle)
        ax.set_xticks(range(n))
        ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        if ylim is not None:
            ax.set_ylim(*ylim)

        for i, v in enumerate(y):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

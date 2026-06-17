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
    s = str(name)
    return s[3:] if s.startswith("b8_") else s


# ---------- resumo por membro (3 KPIs) ----------

def plot_member_summary(
    metrics_by_house: pd.DataFrame,
    outpath,
    *,
    title: str = "",
    community_id: str = "_COMMUNITY",
) -> None:
    """
    3 paineis lado a lado: Excedente (%), Custo liquido (EUR), AutoConsumo (%).
    Um bar por membro + barra da comunidade em destaque (laranja).

    metrics_by_house  DataFrame com indice = house_id e colunas de metricas.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if metrics_by_house.empty:
        return

    members  = [h for h in metrics_by_house.index if h != community_id]
    has_comm = community_id in metrics_by_house.index
    all_ids  = members + ([community_id] if has_comm else [])

    if not all_ids:
        return

    def _val(house_id, col):
        if col not in metrics_by_house.columns:
            return float("nan")
        return float(pd.to_numeric(metrics_by_house.loc[house_id, col], errors="coerce"))

    excedente_pct = []
    custo_eur     = []
    autoconsumo   = []

    for h in all_ids:
        e_pv  = _val(h, "E_pv_kWh")
        e_exp = _val(h, "E_exp_kWh")
        exc   = (e_exp / e_pv * 100.0) if (e_pv and e_pv > 0) else float("nan")
        excedente_pct.append(exc)
        custo_eur.append(_val(h, "Cost_total_EUR"))
        sc = _val(h, "Self_Consumption")
        autoconsumo.append(sc * 100.0 if not np.isnan(sc) else float("nan"))

    labels = [h if h != community_id else "TOTAL" for h in all_ids]
    x      = np.arange(len(all_ids))
    colors = ["#5b9bd5"] * len(members) + (["#ed7d31"] if has_comm else [])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title or "Analise por membro", fontsize=12)

    def _bar_panel(ax, values, ylabel, ylim=None, fmt=".1f"):
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, axis="y")
        if ylim:
            ax.set_ylim(*ylim)
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:{fmt}}", ha="center", va="bottom", fontsize=7)

    ax1.set_title("Excedente PV exportado")
    _bar_panel(ax1, excedente_pct, "% do PV gerado", ylim=(0, 100))

    ax2.set_title("Custo liquido de energia")
    _bar_panel(ax2, custo_eur, "EUR", fmt=".2f")

    ax3.set_title("AutoConsumo medio")
    _bar_panel(ax3, autoconsumo, "% do PV gerado", ylim=(0, 100))

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------- timeseries comparisons across cases ----------

def plot_compare_timeseries(
    dfs_by_case: Dict[str, pd.DataFrame],
    *,
    variable: str,
    outpath,
    title: str = "",
    dt_hours_by_case: Dict[str, float] | None = None,
) -> None:
    """Compare a single variable across cases."""
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
    outpath,
    title: str = "",
    sort_by_value: bool = False,
    higher_is_better: Optional[bool] = None,
) -> None:
    """Bar chart of a single metric across cases."""
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

    if sort_by_value and higher_is_better is not None and n >= 2:
        best  = int(np.argmax(y) if higher_is_better else np.argmin(y))
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
    outpath,
    *,
    title: str = "",
    dt_hours: float = 1.0,
    exclude_keys: tuple = (_COMMUNITY_ID,),
) -> None:
    """Stacked-area chart of alpha_i(t) = PV_i(t) / sum_j PV_j(t) across houses."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    real_houses = [(h, df) for h, df in house_dfs.items() if h not in exclude_keys]
    if not real_houses:
        return

    min_len = min(len(df) for _, df in real_houses)
    pv_matrix = []
    house_labels = []
    t_axis = None

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

    pv_arr   = np.vstack(pv_matrix)
    pv_total = pv_arr.sum(axis=0)
    n_h      = pv_arr.shape[0]
    eq       = 1.0 / n_h
    alpha    = np.where(pv_total > 0, pv_arr / np.where(pv_total > 0, pv_total, 1.0), eq)

    if t_axis is None:
        t_axis = np.arange(pv_arr.shape[1]) * float(dt_hours)
    x_days = t_axis / 24.0

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
    ax_top.stackplot(x_days, alpha, labels=house_labels, alpha=0.85)
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("alpha_i(t)")
    ax_top.set_title(title or "Alocacao PV alpha(t)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right", ncol=min(4, n_h), fontsize=8)

    ax_bot.plot(x_days, pv_total, color="#888888", linewidth=1.0)
    ax_bot.fill_between(x_days, 0, pv_total, color="#cccccc", alpha=0.5)
    ax_bot.set_ylabel("PV total (kW)")
    ax_bot.set_xlabel("tempo (dias)")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------- one-page summary dashboard ----------

def plot_summary_dashboard(
    metrics_df: pd.DataFrame,
    outpath,
    *,
    title: str = "",
    sort_by: str = "Cost_total_EUR",
) -> None:
    """Dashboard comunitario: 3 paineis (SS, SC, Custo) comparando cenarios."""
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
    n     = len(cases)

    panels = [
        ("Self_Sufficiency_COMM", "Auto-suficiencia (comunidade)", "fracao", (0, 1)),
        ("Self_Consumption_COMM", "AutoConsumo (comunidade)",      "fracao", (0, 1)),
        ("Cost_total_EUR",        "Custo total comunidade",        "EUR",    None),
    ]

    fig_w = max(11.0, 0.9 * n + 6.0)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 5))

    for ax, (col, subtitle, ylabel, ylim) in zip(axes, panels):
        if col not in comm.columns:
            ax.text(0.5, 0.5, f"{col}\nnao disponivel", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(subtitle)
            continue

        y    = pd.to_numeric(comm[col], errors="coerce").fillna(0.0).to_numpy()
        bars = ax.bar(range(n), y)

        if n >= 2:
            higher = col in ("Self_Sufficiency_COMM", "Self_Consumption_COMM")
            best   = int(np.argmax(y) if higher else np.argmin(y))
            worst  = int(np.argmin(y) if higher else np.argmax(y))
            bars[best].set_color("#2ca02c")
            bars[worst].set_color("#d62728")

        ax.set_title(subtitle)
        ax.set_xticks(range(n))
        ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        if ylim:
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


# ---------- sweep curve ----------

_SWEEP_DELIM = "__"


def _split_sweep_name(case_name: str) -> tuple:
    if _SWEEP_DELIM in case_name:
        base, _, suffix = case_name.partition(_SWEEP_DELIM)
        return base, suffix
    return case_name, ""


def plot_sweep_curve(
    metrics_df: pd.DataFrame,
    outpath,
    *,
    metric: str,
    title: str = "",
    house: str = _COMMUNITY_ID,
    higher_is_better: Optional[bool] = None,
) -> None:
    """Sweep: x = cenario base, y = metrica, uma linha por variante."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if metrics_df.empty or metric not in metrics_df.columns:
        return

    try:
        sub = metrics_df.xs(house, level="house").copy()
    except KeyError:
        return

    decomposed = [_split_sweep_name(str(c)) for c in sub.index.astype(str)]
    sub = sub.assign(_base=[b for b, _ in decomposed],
                     _suffix=[s for _, s in decomposed])

    bases    = list(dict.fromkeys(sub["_base"].tolist()))
    suffixes = list(dict.fromkeys(sub["_suffix"].tolist()))

    if len(bases) < 2 and len(suffixes) < 2:
        return

    fig_w = max(10.0, 1.0 * len(bases) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    base_labels = [_short_case(b) for b in bases]
    x = np.arange(len(bases))

    for suf in suffixes:
        ys = []
        for b in bases:
            mask = (sub["_base"] == b) & (sub["_suffix"] == suf)
            row  = sub[mask]
            if row.empty:
                ys.append(float("nan"))
            else:
                ys.append(float(pd.to_numeric(row[metric], errors="coerce").iloc[0]))
        label = suf if suf else "(base)"
        ax.plot(x, ys, marker="o", linewidth=2.0, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(base_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_xlabel("cenario base")
    ax.set_title(title or f"Sweep: {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend(title="variante")

    if higher_is_better is not None:
        all_y = pd.to_numeric(sub[metric], errors="coerce").to_numpy()
        if all_y.size > 0 and not np.all(np.isnan(all_y)):
            best_v  = float(np.nanmax(all_y) if higher_is_better else np.nanmin(all_y))
            worst_v = float(np.nanmin(all_y) if higher_is_better else np.nanmax(all_y))
            ax.axhline(best_v,  color="#2ca02c", linestyle=":", alpha=0.6)
            ax.axhline(worst_v, color="#d62728", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------- scatter: cost vs. a community metric ----------

def plot_scatter_cost_vs_metric(
    metrics_df: pd.DataFrame,
    outpath,
    *,
    x_metric: str = "Cost_total_EUR",
    y_metric: str = "Self_Sufficiency_COMM",
    house: str = _COMMUNITY_ID,
    title: str = "",
    annotate: bool = True,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> None:
    """Scatter: trade-off entre x_metric e y_metric, um ponto por cenario."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if metrics_df.empty:
        return

    try:
        sub = metrics_df.xs(house, level="house").copy()
    except KeyError:
        if "house" in metrics_df.columns:
            sub = metrics_df[metrics_df["house"] == house].copy()
        else:
            return

    missing = [m for m in (x_metric, y_metric) if m not in sub.columns]
    if missing:
        return

    x      = pd.to_numeric(sub[x_metric], errors="coerce").to_numpy()
    y      = pd.to_numeric(sub[y_metric], errors="coerce").to_numpy()
    labels = [_short_case(str(c)) for c in sub.index.astype(str)]
    valid  = ~(np.isnan(x) | np.isnan(y))

    if valid.sum() < 2:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x[valid], y[valid], s=80, zorder=3)

    if annotate:
        for xi, yi, lbl in zip(x[valid], y[valid], np.array(labels)[valid]):
            ax.annotate(lbl, (xi, yi), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, clip_on=True)

    ax.set_xlabel(x_label or x_metric)
    ax.set_ylabel(y_label or y_metric)
    ax.set_title(title or f"{y_metric} vs {x_metric}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

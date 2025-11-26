from __future__ import annotations
"""
Relatórios gráficos "completos" para o runner de alphas.

Interface pública (chamar a partir de run_alpha_analysis.py):

    from viz_report import generate_full_report
    generate_full_report(
        out_dir,
        exp_name,
        timestamps,          # pandas.DatetimeIndex ou RangeIndex
        agents,              # list[str]
        load_df,             # DataFrame [t x agents] (kWh por passo)
        pv_df,               # DataFrame [t x ["PV"] ou [t x generators]]; será somado
        alloc_by_cfg,        # dict[str, DataFrame [t x agents]] alocação PV -> agente
        metrics_by_cfg=None, # Optional: DataFrame [cfg x métricas]
        alpha_by_cfg=None,   # Optional: dict[str, DataFrame [t x agents]] coeficientes
        receiver_mask=None,  # Optional: DataFrame [t x agents] (True/False)
        dpi=144,
        img_format="png"
    )

Cria um conjunto de figuras com visão global, comparações entre configurações e painéis por agente.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
sns.set_style("whitegrid")

# -----------------------------
# Helpers
# -----------------------------

def _ensure_outdir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sum_pv(pv_df: pd.DataFrame) -> pd.Series:
    cols = [c for c in pv_df.columns if str(c).lower().startswith(("pv","gen","prod"))] or list(pv_df.columns)
    return pv_df[cols].sum(axis=1)


def _safe_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.copy()
        for m in missing:
            df[m] = 0.0
    return df[cols]


def _align_df_to_t(df: pd.DataFrame, t: pd.Index) -> pd.DataFrame:
    """Alinha por índice; se faltar algo, preenche a 0 (posicionalmente seguro)."""
    if not df.index.equals(t):
        df = df.reindex(t, fill_value=0.0)
    return df


def _align_s_to_t(s: pd.Series, t: pd.Index) -> pd.Series:
    if not s.index.equals(t):
        s = s.reindex(t, fill_value=0.0)
    return s


# -----------------------------
# Core plots
# -----------------------------

def plot_overview(axs, t: pd.Index, community_pv: pd.Series, load_df: pd.DataFrame,
                  alloc_by_cfg: Dict[str, pd.DataFrame]):
    """Desenha 3x2: PV vs Load, Balanço, Duração, Métricas simples, etc."""
    # Alinhamento defensivo
    load_df = _align_df_to_t(load_df, t)
    community_pv = _align_s_to_t(community_pv, t)
    alloc_by_cfg = {name: _align_df_to_t(df, t) for name, df in alloc_by_cfg.items()}

    agents = list(load_df.columns)

    # 1) PV total vs Load total
    ax = axs[0,0]
    ax.plot(t, community_pv.values, label="PV", color="#f59e0b")
    ax.plot(t, load_df.sum(axis=1).values, label="Carga total", color="#2563eb")
    ax.set_title("PV vs Carga (total)")
    ax.set_ylabel("kWh/intervalo")
    ax.legend(loc="upper right")

    # 2) Balanço comunitário para 1ª config
    first_cfg, first_alloc = next(iter(alloc_by_cfg.items()))
    ax = axs[0,1]
    alloc_tot = first_alloc.sum(axis=1).values
    import_series = (load_df.sum(axis=1).values - alloc_tot).clip(min=0)
    export_series = (community_pv.values - alloc_tot).clip(min=0)
    ax.stackplot(t, alloc_tot, import_series, export_series,
                 labels=[f"Alocado ({first_cfg})", "Import", "Export"],
                 colors=["#10b981", "#ef4444", "#6b7280"], alpha=0.9)
    ax.set_title("Balanço comunitário (amostra)")
    ax.legend(loc="upper right")

    # 3) Curvas de duração (Import)
    ax = axs[1,0]
    for name, A in alloc_by_cfg.items():
        import_series = (load_df.sum(axis=1) - A.sum(axis=1)).clip(lower=0)
        y = np.sort(import_series.values)[::-1]
        ax.plot(y, label=name)
    ax.set_title("Curva de duração do Import")
    ax.set_ylabel("kWh/intervalo")
    ax.legend()

    # 3b) Curvas de duração (Export)
    ax = axs[1,1]
    for name, A in alloc_by_cfg.items():
        export_series = (community_pv - A.sum(axis=1)).clip(lower=0)
        y = np.sort(export_series.values)[::-1]
        ax.plot(y, label=name)
    ax.set_title("Curva de duração do Export")
    ax.legend()

    # 4) Stack por agente (config amostra)
    ax = axs[2,0]
    first_alloc = _safe_cols(first_alloc, agents)
    ax.stackplot(t, *[first_alloc[c].values for c in agents], labels=agents)
    ax.set_title(f"Alocação por agente – {first_cfg}")
    ax.set_ylabel("kWh/intervalo")
    if len(agents) <= 15:
        ax.legend(loc="upper right", ncols=2, fontsize="small")

    # 5) Comparação simples: PV comunitário utilizado por cfg
    ax = axs[2,1]
    totals = {}
    for name, A in alloc_by_cfg.items():
        totals[name] = (A.sum(axis=1).clip(upper=community_pv)).sum()
    names, vals = zip(*sorted(totals.items(), key=lambda kv: -kv[1]))
    ax.barh(names, vals, color="#0ea5e9")
    ax.set_title("PV utilizado na comunidade (kWh)")
    ax.invert_yaxis()


def plot_alpha_heatmaps(out_dir: Path, t: pd.Index, alpha_by_cfg: Dict[str, pd.DataFrame],
                        receiver_mask: Optional[pd.DataFrame] = None,
                        dpi: int = 144, img_format: str = "png"):
    if not alpha_by_cfg:
        return
    for name, A in alpha_by_cfg.items():
        data = _align_df_to_t(A, t).copy()
        if receiver_mask is not None:
            mask = _align_df_to_t(receiver_mask, t)
            # Reindex columns to match before masking (evita broadcast errado)
            mask = mask.reindex(columns=data.columns, fill_value=True)
            data = data.where(mask, other=0.0)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        sns.heatmap(data.T, cmap="crest", ax=ax, cbar_kws={"label": "α"})
        ax.set_title(f"Mapa de calor dos coeficientes – {name}")
        ax.set_xlabel("tempo")
        ax.set_ylabel("agente")
        fig.tight_layout()
        fig.savefig(out_dir / f"alpha_heatmap__{name}.{img_format}", dpi=dpi)
        plt.close(fig)


def plot_per_agent_panels(out_dir: Path, t: pd.Index, agent: str,
                           load: pd.Series, pv: pd.Series,
                           alloc_by_cfg: Dict[str, pd.DataFrame],
                           dpi: int = 144, img_format: str = "png"):
    # Alinhamento defensivo
    load = _align_s_to_t(load, t)
    pv   = _align_s_to_t(pv, t)
    alloc_by_cfg = {name: _align_df_to_t(df, t) for name, df in alloc_by_cfg.items()}

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    # 1) Load vs PV (quota comunitária disponível)
    axs[0,0].plot(t, load.values, label="Load", color="#1f77b4")
    axs[0,0].plot(t, pv.values, label="PV total", color="#ff7f0e", alpha=0.6)
    axs[0,0].set_title(f"{agent}: carga vs PV comunitário")
    axs[0,0].legend()

    # 2) Alocação por configuração
    for name, A in alloc_by_cfg.items():
        s = A[agent] if agent in A.columns else pd.Series(0.0, index=A.index)
        axs[0,1].plot(t, s.values, label=name)
    axs[0,1].set_title("Energia alocada por configuração")
    axs[0,1].legend(fontsize="small")

    # 3) Import líquido por configuração
    for name, A in alloc_by_cfg.items():
        s_alloc = A[agent] if agent in A.columns else pd.Series(0.0, index=A.index)
        net_import = (load - s_alloc).clip(lower=0)
        axs[1,0].plot(t, net_import.values, label=name)
    axs[1,0].set_title("Import líquido")
    axs[1,0].legend(fontsize="small")

    # 4) Curva de duração do import
    for name, A in alloc_by_cfg.items():
        s_alloc = A[agent] if agent in A.columns else pd.Series(0.0, index=A.index)
        y = np.sort((load - s_alloc).clip(lower=0).values)[::-1]
        axs[1,1].plot(y, label=name)
    axs[1,1].set_title("Duração do import")
    axs[1,1].legend(fontsize="small")

    fig.suptitle(f"Painel do agente {agent}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"agent_panel__{agent}.{img_format}", dpi=dpi)
    plt.close(fig)


def plot_metrics_bar(out_dir: Path, metrics_by_cfg: pd.DataFrame,
                     img_format: str = "png", dpi: int = 144):
    if metrics_by_cfg is None or metrics_by_cfg.empty:
        return
    df = metrics_by_cfg.copy()
    order = list(df.index)
    fig, axs = plt.subplots(1, len(df.columns), figsize=(4*len(df.columns), 4))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    for ax, col in zip(axs, df.columns):
        ax.bar(order, df[col].values, color="#22c55e")
        ax.set_title(col)
        ax.tick_params(axis='x', rotation=45)
    fig.suptitle("Comparação de métricas por configuração")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"metrics_comparison.{img_format}", dpi=dpi)
    plt.close(fig)




def plot_total_alloc_vs_used_by_agent(out_dir: Path, load_df: pd.DataFrame,
                                      alloc_by_cfg: Dict[str, pd.DataFrame],
                                      img_format: str = "png", dpi: int = 144) -> None:
    """
    Para cada configuração, cria um gráfico comparativo (barras lado a lado)
    com o TOTAL **disponível** vs TOTAL **usado** por agente, no horizonte.
    - disponível = soma_t alloc[a,t]
    - usado      = soma_t min(alloc[a,t], load[a,t])
    """
    for name, A in alloc_by_cfg.items():
        A = _align_df_to_t(A, load_df.index)
        U = pd.DataFrame(np.minimum(A.values, load_df.values), index=A.index, columns=A.columns)
        tot_alloc = A.sum(axis=0)
        tot_used  = U.sum(axis=0)

        # Ordena por total alocado (desc) para facilitar leitura
        order = tot_alloc.sort_values(ascending=False).index
        x = np.arange(len(order))
        w = 0.42

        fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.5*len(order)), 6))
        ax.bar(x - w/2, tot_alloc.loc[order].values, width=w, label="Total disponível")
        ax.bar(x + w/2, tot_used.loc[order].values,  width=w, label="Total usado")

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel("kWh (total no horizonte)")
        ax.set_title(f"Total vs Usado por agente – {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"total_vs_usado_por_agente__{name}.{img_format}", dpi=dpi)
        plt.close(fig)
# -----------------------------
# Public API
# -----------------------------

def generate_full_report(
    out_dir: str | os.PathLike,
    exp_name: str,
    timestamps: pd.Index,
    agents: List[str],
    load_df: pd.DataFrame,
    pv_df: pd.DataFrame,
    alloc_by_cfg: Dict[str, pd.DataFrame],
    metrics_by_cfg: Optional[pd.DataFrame] = None,
    alpha_by_cfg: Optional[Dict[str, pd.DataFrame]] = None,
    receiver_mask: Optional[pd.DataFrame] = None,
    dpi: int = 144,
    img_format: str = "png",
):
    out_dir = _ensure_outdir(out_dir)

    # 0) Sanidade: mesmo número de linhas em tudo
    H = len(timestamps)
    if len(load_df) != H:
        raise ValueError(f"load_df rows ({len(load_df)}) != timeline rows ({H})")
    if len(pv_df) != H:
        raise ValueError(f"pv_df rows ({len(pv_df)}) != timeline rows ({H})")
    for name, A in alloc_by_cfg.items():
        if len(A) != H:
            raise ValueError(f"alloc_by_cfg['{name}'] rows ({len(A)}) != timeline rows ({H})")
    if alpha_by_cfg is not None:
        for name, A in alpha_by_cfg.items():
            if len(A) != H:
                raise ValueError(f"alpha_by_cfg['{name}'] rows ({len(A)}) != timeline rows ({H})")
    if receiver_mask is not None and len(receiver_mask) != H:
        raise ValueError(f"receiver_mask rows ({len(receiver_mask)}) != timeline rows ({H})")

    # 1) Alinhar tudo ao mesmo índice 'timestamps'
    load_df = _align_df_to_t(load_df, timestamps)
    pv_df   = _align_df_to_t(pv_df, timestamps)
    alloc_by_cfg = {k: _align_df_to_t(v, timestamps) for k, v in alloc_by_cfg.items()}
    if alpha_by_cfg is not None:
        alpha_by_cfg = {k: _align_df_to_t(v, timestamps) for k, v in alpha_by_cfg.items()}
    if receiver_mask is not None:
        receiver_mask = _align_df_to_t(receiver_mask, timestamps)

    # 2) Overview 3x2
    pv_total = _sum_pv(pv_df)
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    plot_overview(axs, timestamps, pv_total, load_df, alloc_by_cfg)
    fig.suptitle(f"Relatório – {exp_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"overview__{exp_name}.{img_format}", dpi=dpi)
    plt.close(fig)

    # 2c) Comparação: Total disponível vs Total usado por agente (por configuração)
    plot_total_alloc_vs_used_by_agent(out_dir, load_df, alloc_by_cfg, img_format, dpi)

    # 3) Heatmaps de alphas
    plot_alpha_heatmaps(out_dir, timestamps, alpha_by_cfg or {}, receiver_mask, dpi, img_format)

    # 4) Painéis por agente
    for ag in agents:
        plot_per_agent_panels(out_dir, timestamps, ag, load_df[ag], pv_total, alloc_by_cfg, dpi, img_format)

    # 5) Barras de métricas
    if metrics_by_cfg is not None:
        plot_metrics_bar(out_dir, metrics_by_cfg, img_format, dpi)
    # 5c) Checagem de sanidade: soma(alloc) == PV_total por passo
    pv_total = _sum_pv(pv_df)
    for name, A in alloc_by_cfg.items():
        diff = (A.sum(axis=1) - pv_total).abs().max()
        if diff > 1e-6:
            print(f"[WARN] {name}: soma(alloc) != PV_total; max|dif| = {diff:.3g}")



    # 6) CSVs auxiliares (auditoria)
    audit = pd.DataFrame({
        "PV_total": _sum_pv(pv_df),
        "Load_total": load_df.sum(axis=1),
    }, index=timestamps)
    for name, A in alloc_by_cfg.items():
        audit[f"Alloc_total__{name}"] = A.sum(axis=1)
    audit.to_csv(out_dir / f"audit_timeseries__{exp_name}.csv")

    if metrics_by_cfg is not None:
        metrics_by_cfg.to_csv(out_dir / f"metrics__{exp_name}.csv")

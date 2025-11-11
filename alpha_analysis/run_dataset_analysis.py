from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_N_AGENTS: int = 3
DEFAULT_OUT_DIR: str = "dataset report"

def read_headerless_ignore_first(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {path}")
    # Ignora logo a 1ª coluna na leitura (menos memória/cópias)
    df = pd.read_csv(path, header=None, sep=None, engine="python", usecols=lambda i: i > 0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    logging.debug("Lido %s -> shape=%s", path.name, df.shape)
    return df

def build_time_index(n_rows: int, start_date: str, freq: str) -> pd.DatetimeIndex:
    """Índice temporal naive (sem timezone)."""
    return pd.date_range(start=pd.to_datetime(start_date), periods=n_rows, freq=freq)

def logical_month_index(idx: pd.DatetimeIndex) -> pd.Series:
    """Mês lógico M0, M1, ... (compatível com todas as versões de pandas)."""
    y0, m0 = idx[0].year, idx[0].month
    return (idx.year - y0) * 12 + (idx.month - m0)


def month_boundaries(idx: pd.DatetimeIndex) -> tuple[list[pd.Timestamp], list[str]]:
    """Inícios de cada mês lógico e labels 'M0', 'M1', ...'"""
    m_idx = logical_month_index(idx).to_numpy()
    change_pos = np.r_[0, np.flatnonzero(np.diff(m_idx)) + 1]
    starts = [idx[p] for p in change_pos.tolist()]
    labels = [f"M{m_idx[p]}" for p in change_pos.tolist()]
    return starts, labels


def align_shapes(load: pd.DataFrame, pv: pd.DataFrame, n_agents: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    n = min(n_agents, load.shape[1], pv.shape[1])
    if n == 0:
        raise ValueError("Não há colunas suficientes para alinhar agentes após ignorar a 1ª coluna.")
    agents = [f"agent{i}" for i in range(1, n + 1)]

    load = load.iloc[:, :n].copy()
    pv   = pv.iloc[:, :n].copy()
    load.columns = agents
    pv.columns   = agents

    T = min(len(load), len(pv))
    load = load.iloc[:T]
    pv   = pv.iloc[:T]

    logging.debug("Agentes: %s | Horizonte: %d pontos", agents, T)
    return load, pv, agents, T

# ==========================
# Agregação mensal lógica (global)
# ==========================
def global_monthly_table(load: pd.DataFrame, pv: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    load = load.set_index(index)
    pv   = pv.set_index(index)

    df = pd.DataFrame({
        "month_id": logical_month_index(index).values,
        "load": load.sum(axis=1).values,
        "pv":   pv.sum(axis=1).values,
    })
    agg = (df.groupby("month_id", as_index=False)
             .sum()
             .rename(columns={"load": "load_total", "pv": "pv_total"}))
    logging.debug("Tabela global mensal: %d linhas", len(agg))
    return agg

def plot_global_monthly(agg: pd.DataFrame, outdir: Path) -> Path:
    """Barras PV vs Consumo"""
    outdir.mkdir(parents=True, exist_ok=True)

    x_ids = agg["month_id"].to_numpy()
    labels = [f"M{m}" for m in x_ids]
    x = np.arange(len(x_ids))
    width = 0.4

    fig, ax = plt.subplots()
    ax.bar(x - width/2, agg["pv_total"].to_numpy(), width, label="PV Total (mês lógico)")
    ax.bar(x + width/2, agg["load_total"].to_numpy(), width, label="Consumo Total (mês lógico)")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Mês lógico")
    ax.set_ylabel("Total (unid. orig.)")
    ax.set_title("Global por mês lógico (PV vs Consumo)")
    ax.legend()
    fig.tight_layout()

    out_path = outdir / "global_mensal.png"
    fig.savefig(out_path, dpi=144)
    plt.close(fig)
    logging.info("Gráfico global mensal guardado em: %s", out_path)
    return out_path

def plot_per_agent(load: pd.DataFrame, pv: pd.DataFrame, agents: list[str], index: pd.DatetimeIndex, outdir: Path) -> list[Path]:
    """Consumo vs PV por agente"""
    outdir.mkdir(parents=True, exist_ok=True)
    load = load.set_index(index)
    pv   = pv.set_index(index)

    starts, labels = month_boundaries(index)
    outputs: list[Path] = []

    for ag in agents:
        fig, ax = plt.subplots()
        ax.plot(load.index, load[ag], label="Consumo")
        ax.plot(pv.index,   pv[ag],   label="PV")

        for s in starts:
            ax.axvline(s, linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_xticks(starts)
        ax.set_xticklabels(labels)
        ax.set_title(f"{ag} — Consumo vs PV (meses lógicos)")
        ax.set_xlabel("Mês lógico")
        ax.set_ylabel("Potência / Energia (unid. orig.)")
        ax.legend()
        fig.tight_layout()

        out_path = outdir / f"agent_{ag}.png"
        fig.savefig(out_path, dpi=144)
        plt.close(fig)
        outputs.append(out_path)

    logging.info("Gráficos por agente guardados em: %s", outdir)
    return outputs

DEFAULT_START_DATE: str = "2000-01-01 00:00"
DEFAULT_FREQ: str = "15min"

# ==========================
# Run / Main
# ==========================
def run(
    data_dir: str = "Data",
    out_dir: str = DEFAULT_OUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    freq: str = DEFAULT_FREQ,
    n_agents: int = DEFAULT_N_AGENTS,
) -> None:
    # Configura logging aqui (menos uma função no módulo)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    data_dir_p = Path(data_dir)
    out_dir_p = Path(out_dir)

    load_raw = read_headerless_ignore_first(data_dir_p / "load_cons.csv")
    pv_raw   = read_headerless_ignore_first(data_dir_p / "pv_gen.csv")

    load, pv, agents, T = align_shapes(load_raw, pv_raw, n_agents=n_agents)
    idx = build_time_index(T, start_date=start_date, freq=freq)

    agg = global_monthly_table(load, pv, idx)
    global_path = plot_global_monthly(agg, out_dir_p / "figs")
    agent_paths = plot_per_agent(load, pv, agents, idx, out_dir_p / "figs")

    m_max = int(logical_month_index(idx).max())
    print(f"Gráfico global mensal: {global_path}")
    print(f"Figuras por agente ({len(agent_paths)}): {out_dir_p / 'figs'}")
    print("Agentes:", ", ".join(agents))
    print(f"Período lógico: M0 → M{m_max} ({len(idx)} pontos @ {freq})")

def main() -> None:
    run()

if __name__ == "__main__":
    main()

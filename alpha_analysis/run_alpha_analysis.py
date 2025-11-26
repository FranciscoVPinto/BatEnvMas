from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Sequence, Any
import pandas as pd
import numpy as np

from analysis.alphas import (
    alpha_equal,
    alpha_proportional_mean_load,
    alpha_instant_load_share,
    normalize_alpha_cols,
    alpha_fixed_from_shares,
    alpha_hierarchical,
    alpha_dynamic_windows,
)
from analysis.allocate import allocate_unlimited
from analysis.metrics import energy_kpis, fairness_index, jains_index
from analysis.viz_report import generate_full_report

from analysis.helpers import (
    ensure_outdir, save_csvs,
    load_wide_csv_with_ts, take_first_cols, slice_week,
    remap_shares, remap_groups, remap_windows,
)

# ================= USER SETTINGS =================
LOAD_CSV = Path("dataset/load_cons.csv")
PV_CSV   = Path("dataset/pv_gen.csv")

SKIP_FIRST_COLS = 1
MAX_LOAD_COLS   = 3
MAX_PV_COLS     = 3

INTERVAL_MIN    = 15
WEEK            = 1
WEEK_OFFSET     = 0
HORIZON         = 0

# Alphas a correr (mistura livres e com parâmetros)
ALPHAS_SPECS: Sequence[Dict[str, Any]] = (
    {"kind": "equal"},
    {"kind": "mean_load"},
    {"kind": "instant_load"},
    {"kind": "fixed_shares", "shares": {"A1": 0.5, "A2": 0.3, "A3": 0.2}},
    {
        "kind": "hierarchical",
        "groups": {"G1": ["A1", "A2"], "G2": ["A3"]},
        "inner_rule": "InstantLoad",   # "Equal" | "ProportionalMean" | "InstantLoad"
        "outer_shares": {"G1": 0.6, "G2": 0.4},
    },
    {
        "kind": "dynamic_windows",
        "windows": [("START", "MID", {"A1": 0.7, "A2": 0.3, "A3": 0.0}),
                    ("MID",   "END", {"A1": 0.2, "A2": 0.4, "A3": 0.4})],
    },
)

OUTDIR          = Path("outputs/alpha_run")
SAVE            = True

# Visual report
GENERATE_REPORT = True
EXP_NAME        = "alpha_run"
REPORT_DPI      = 144
REPORT_FORMAT   = "png"
# ==================================================


@dataclass
class AlphaRunConfig:
    load_csv: Path
    pv_csv: Path
    skip_first_cols: int = 1
    max_load_cols: Optional[int] = 3
    max_pv_cols: Optional[int] = 3
    interval_min: int = 15
    week: int = 1
    week_offset: int = 0
    horizon: int = 0
    alpha_specs: Sequence[Dict[str, Any]] = ()
    outdir: Path = Path("outputs")
    save: bool = False
    generate_report: bool = False
    exp_name: str = "run"
    report_dpi: int = 144
    report_format: str = "png"


def _build_alpha_from_spec(spec: Dict[str, Any], loads: pd.DataFrame, agents: List[str], H: int) -> pd.DataFrame:
    """Converte um spec {'kind':..., ...} numa matriz alpha (n, H) normalizada."""
    kind = spec.get("kind", "").strip().lower()

    if kind == "equal":
        A = alpha_equal(len(agents), H)

    elif kind in ("mean_load", "proportional_mean"):
        A = alpha_proportional_mean_load(loads)

    elif kind in ("instant_load", "instant"):
        A = alpha_instant_load_share(loads)

    elif kind in ("fixed_shares", "fixed"):
        shares = spec.get("shares")
        if not isinstance(shares, dict):
            raise ValueError("fixed_shares requer dict 'shares' {agent: peso}.")
        shares = remap_shares(shares, agents)
        A = alpha_fixed_from_shares(shares, agents, H)

    elif kind == "hierarchical":
        groups = spec.get("groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("hierarchical requer 'groups' = {grupo: [agents...]}.")
        inner_rule = spec.get("inner_rule", "InstantLoad")
        outer_shares = spec.get("outer_shares")
        if isinstance(outer_shares, dict):
            outer_shares = {g: float(v) for g, v in outer_shares.items()}
        groups = remap_groups(groups, agents)
        A = alpha_hierarchical(groups=groups, agents=agents, H=H,
                               inner_rule=inner_rule, outer_shares=outer_shares, load_df=loads)

    elif kind in ("dynamic_windows", "dynamic"):
        windows = spec.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("dynamic_windows requer 'windows' = [(t0, t1_excl, {agent: peso}), ...].")
        windows = remap_windows(windows, agents, H)
        A = alpha_dynamic_windows(agents, H, windows)

    else:
        raise ValueError(f"Unknown alpha kind: {spec.get('kind')}")

    return normalize_alpha_cols(A)


# ---------------- Core run ----------------

def run(cfg: AlphaRunConfig):
    # Load
    load_df, _ = load_wide_csv_with_ts(cfg.load_csv, cfg.skip_first_cols)
    pv_df, _   = load_wide_csv_with_ts(cfg.pv_csv,   cfg.skip_first_cols)

    # Keep only first N columns
    load_df = take_first_cols(load_df, cfg.max_load_cols)
    pv_df   = take_first_cols(pv_df,   cfg.max_pv_cols)

    # PV community series
    pv_series = pv_df.sum(axis=1)

    # Slice (positional)
    load_df   = slice_week(load_df,  cfg.interval_min, cfg.week, cfg.week_offset, cfg.horizon)
    pv_df     = slice_week(pv_df,    cfg.interval_min, cfg.week, cfg.week_offset, cfg.horizon)
    pv_series = slice_week(pv_series.to_frame("PV"), cfg.interval_min, cfg.week, cfg.week_offset, cfg.horizon)["PV"]

    # Align RangeIndex
    H = len(load_df)
    if not (H == len(pv_df) == len(pv_series)):
        raise ValueError(f"Length mismatch after slicing: load={len(load_df)}, pv_df={len(pv_df)}, pv_series={len(pv_series)}")
    load_df = load_df.reset_index(drop=True)
    pv_df   = pv_df.reset_index(drop=True)
    pv_series.index = pd.RangeIndex(H)
    timestamps = pd.RangeIndex(H)

    # Agents
    agents: List[str] = list(load_df.columns)

    # ---- Multi-alpha run ----
    alloc_by_cfg: Dict[str, pd.DataFrame] = {}
    alpha_by_cfg: Dict[str, pd.DataFrame] = {}
    metrics_rows: List[pd.DataFrame] = []

    for spec in cfg.alpha_specs:
        kind_name = spec.get("kind", "unknown")
        key = kind_name  # nome da config no report/CSV

        # Build alpha (n, H)
        A = _build_alpha_from_spec(spec, loads=load_df, agents=agents, H=H)

        # Allocation (H x n)
        alloc_df = allocate_unlimited(pv_series, A, agents)
        alloc_df.index = timestamps

        # Usado (cortado pela carga)
        used_df = pd.DataFrame(
            np.minimum(alloc_df.values, load_df.values),
            index=alloc_df.index, columns=alloc_df.columns
        )

        # KPIs + fairness
        base_kpis = energy_kpis(load_df, alloc_df, pv_series)  # 1-row DF
        row = base_kpis.iloc[0].to_dict()
        row["Fairness"] = fairness_index(alloc_df, load_df)
        row["Jain"] = jains_index(alloc_df, load_df)
        row["FairnessUsed"] = fairness_index(used_df, load_df)
        row["JainUsed"] = jains_index(used_df, load_df)
        kpis = pd.DataFrame([row], index=[key])
        metrics_rows.append(kpis)

        # guardas p/ relatório
        alloc_by_cfg[key] = alloc_df
        alpha_by_cfg[key] = pd.DataFrame(A.T, index=timestamps, columns=agents)

        # guardar CSVs por-config
        if cfg.save:
            out_k = cfg.outdir / key
            save_csvs(out_k, {
                "load_week.csv": load_df,
                "pv_week.csv": pv_series.to_frame("PV"),
                "allocation_unlimited.csv": alloc_df,
                "allocation_used_capped.csv": used_df,
                "kpis.csv": kpis,
            })

    # Tabela final de métricas
    metrics_by_cfg = pd.concat(metrics_rows, axis=0)

    # Consolidado topo
    if cfg.save:
        ensure_outdir(cfg.outdir)
        metrics_by_cfg.to_csv(cfg.outdir / "metrics_all_configs.csv", index=True)

    # Summary
    rows_per_day = int(24 * 60 // cfg.interval_min)
    rows_per_week = rows_per_day * 7
    print("\n=== Configuration ===")
    print(f"Agents: {len(agents)} | Interval: {cfg.interval_min} min | Horizon rows: {cfg.horizon}")
    print(f"Week: {cfg.week} (offset +{cfg.week_offset}) | Configs: {', '.join([s['kind'] for s in cfg.alpha_specs])}")
    print(f"Rows/day: {rows_per_day} | Rows/week: {rows_per_week}")
    print(f"Load rows after slicing: {len(load_df)} | PV rows after slicing: {len(pv_series)}")
    print("\n=== Energy totals per config ===")
    print(metrics_by_cfg.to_string())

    # Report
    if cfg.generate_report:
        generate_full_report(
            out_dir=cfg.outdir,
            exp_name=cfg.exp_name,
            timestamps=timestamps,
            agents=agents,
            load_df=load_df,
            pv_df=pv_df,
            alloc_by_cfg=alloc_by_cfg,
            metrics_by_cfg=metrics_by_cfg,
            alpha_by_cfg=alpha_by_cfg,
            receiver_mask=None,
            dpi=cfg.report_dpi,
            img_format=cfg.report_format,
        )

    # devolver última config
    last_key = list(alloc_by_cfg.keys())[-1]
    return {
        "kpis": metrics_by_cfg.loc[[last_key]],
        "alloc_df": alloc_by_cfg[last_key],
        "metrics_all": metrics_by_cfg
    }


if __name__ == "__main__":
    cfg = AlphaRunConfig(
        load_csv=Path(LOAD_CSV),
        pv_csv=Path(PV_CSV),
        skip_first_cols=int(SKIP_FIRST_COLS),
        max_load_cols=MAX_LOAD_COLS,
        max_pv_cols=MAX_PV_COLS,
        interval_min=int(INTERVAL_MIN),
        week=int(WEEK),
        week_offset=int(WEEK_OFFSET),
        horizon=int(HORIZON),
        alpha_specs=tuple(ALPHAS_SPECS),
        outdir=Path(OUTDIR),
        save=bool(SAVE),
        generate_report=bool(GENERATE_REPORT),
        exp_name=str(EXP_NAME),
        report_dpi=int(REPORT_DPI),
        report_format=str(REPORT_FORMAT),
    )
    run(cfg)

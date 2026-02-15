from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batEnv.io import load_case_yaml
from batEnv.plotting import (
    compute_summary_metrics,
    plot_house_per_case,
    plot_compare_timeseries,
    plot_compare_metrics,
)
from batEnv.utils.community_metrics import COMMUNITY_ID, aggregate_community_timeseries, compute_community_extra_metrics


def load_plotset_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("plotset YAML must parse to a dict.")
    cfg["_plotset_path"] = str(path.resolve())
    return cfg


def resolve_case_path(cases_base_dir: Path, item: str) -> Path:
    p = Path(item)
    if not p.is_absolute():
        p = (cases_base_dir / p).resolve()
    return p


def get_case_name_and_dt(case_yaml_path: Path) -> Tuple[str, float]:
    c = load_case_yaml(case_yaml_path)
    case_name = c.get("case", case_yaml_path.stem)
    time_cfg = c.get("time", {}) if isinstance(c.get("time", {}), dict) else {}
    dt_hours = float(time_cfg.get("dt_hours", 1.0))
    return str(case_name), dt_hours


def read_house_csvs(case_out_dir: Path) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(case_out_dir.glob("results_house_*.csv")):
        house_id = csv_path.stem.replace("results_house_", "")
        dfs[house_id] = pd.read_csv(csv_path)
    return dfs


def make_per_case_plots(case_name: str, case_out_dir: Path, dt_hours: float, *, include_community: bool = False) -> None:
    plots_dir = case_out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    house_dfs = read_house_csvs(case_out_dir)
    if not house_dfs:
        print(f"[WARN] {case_name}: no results_house_*.csv in {case_out_dir}")
        return

    if include_community:
        df_comm = aggregate_community_timeseries(house_dfs)
        if not df_comm.empty:
            house_dfs = dict(house_dfs)
            house_dfs[COMMUNITY_ID] = df_comm

    for house_id, df in house_dfs.items():
        out_png = plots_dir / f"plot_house_{house_id}.png"
        plot_house_per_case(df, out_png, title=f"{case_name} | house {house_id}", dt_hours=dt_hours)
        print(f"[OK] per-case: {case_name} / house {house_id}")

    print(f"[OK] per-case saved: {plots_dir}")


def make_comparisons(
    plotset_name: str,
    outputs_dir: Path,
    cases_info: list[tuple[str, Path, float]],
) -> None:
    comp_root = outputs_dir / "_comparisons" / plotset_name
    comp_root.mkdir(parents=True, exist_ok=True)

    case_house_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    houses_union = set()
    dt_by_case: Dict[str, float] = {}

    for case_name, case_out_dir, dt_hours in cases_info:
        dt_by_case[case_name] = dt_hours
        if not case_out_dir.exists():
            continue

        hd = read_house_csvs(case_out_dir)

        df_comm = aggregate_community_timeseries(hd)
        if not df_comm.empty:
            hd = dict(hd)
            hd[COMMUNITY_ID] = df_comm

        case_house_dfs[case_name] = hd
        houses_union |= set(hd.keys())

    houses_union = sorted(houses_union)

    metrics_all_list = []
    for case_name, case_out_dir, dt_hours in cases_info:
        if not case_out_dir.exists():
            continue
        hd = case_house_dfs.get(case_name, {})
        for house_id, df in hd.items():
            m = compute_summary_metrics(df, dt_hours=dt_hours)
            if house_id == COMMUNITY_ID:
                m.update(compute_community_extra_metrics(df, dt_hours=dt_hours))
            m["case"] = case_name
            m["house"] = house_id
            metrics_all_list.append(m)

    if metrics_all_list:
        metrics_all = pd.DataFrame(metrics_all_list).set_index(["case", "house"]).sort_index()
        metrics_csv = comp_root / "metrics_all.csv"
        metrics_all.to_csv(metrics_csv)
        print(f"[OK] metrics table: {metrics_csv}")
    else:
        metrics_all = pd.DataFrame()
        print("[WARN] No metrics could be computed (no CSVs found).")

    compare_vars = [
        "E",
        "P_imp",
        "P_exp",
        "P_ch",
        "P_dis",
        "cost_step",
        "cost_cum",
        "P_net_grid",
        "P_simul_imp_exp",
    ]
    ts_root = comp_root / "timeseries"
    ts_root.mkdir(parents=True, exist_ok=True)

    for house_id in houses_union:
        dfs_for_house = {}
        for case_name, _, _ in cases_info:
            df = case_house_dfs.get(case_name, {}).get(house_id)
            if df is not None:
                dfs_for_house[case_name] = df

        if len(dfs_for_house) < 2:
            continue

        for var in compare_vars:
            out_png = ts_root / f"compare_{var}_house_{house_id}.png"
            plot_compare_timeseries(
                dfs_for_house,
                variable=var,
                outpath=out_png,
                title=f"Compare {var} | {house_id}",
                dt_hours_by_case=dt_by_case,
            )
        print(f"[OK] comparisons: timeseries for {house_id}")

    if not metrics_all.empty:
        bar_root = comp_root / "metrics"
        bar_root.mkdir(parents=True, exist_ok=True)

        preferred = [
            "Cost_total_EUR",
            "E_imp_kWh",
            "E_exp_kWh",
            "E_ch_kWh",
            "E_dis_kWh",
            "E_end_kWh",
            "E_simul_imp_exp_kWh",
        ]
        metric_list = [m for m in preferred if m in metrics_all.columns]

        for house_id in houses_union:
            try:
                sub = metrics_all.xs(house_id, level="house")
            except Exception:
                continue
            if sub.shape[0] < 2:
                continue

            for metric in metric_list:
                out_png = bar_root / f"bar_{metric}_house_{house_id}.png"
                plot_compare_metrics(sub, metric=metric, outpath=out_png, title=f"{metric} | {house_id}")

        print(f"[OK] comparisons: metric bars saved to {bar_root}")

    print(f"[OK] comparisons root: {comp_root}")


def make_plots_all(plotset_yaml: str | Path) -> None:
    ps = load_plotset_yaml(plotset_yaml)

    plotset_name = str(ps.get("plotset", Path(plotset_yaml).stem))

    cases_base_dir = Path(ps.get("cases_base_dir", "cases"))
    if not cases_base_dir.is_absolute():
        cases_base_dir = (ROOT / cases_base_dir).resolve()

    outputs_dir = Path(ps.get("outputs_dir", "results"))
    if not outputs_dir.is_absolute():
        outputs_dir = (ROOT / outputs_dir).resolve()

    enabled_map = ps.get("enabled", {}) or {}
    if not isinstance(enabled_map, dict):
        raise ValueError("enabled must be a dict mapping case_name -> true/false")

    plots_cfg = ps.get("plots", {}) if isinstance(ps.get("plots", {}), dict) else {}
    do_per_case = bool(plots_cfg.get("per_case", True))
    do_comparisons = bool(plots_cfg.get("comparisons", True))
    include_comm_per_case = bool(plots_cfg.get("include_community_per_case", False))

    case_items = ps.get("cases", [])
    if not isinstance(case_items, list) or not case_items:
        raise ValueError("plotset.cases must be a non-empty list of case YAML files")

    print(f"[PLOTSET] {plotset_name}")
    print(f"[PLOTSET] cases base dir: {cases_base_dir.resolve()}")
    print(f"[PLOTSET] outputs dir   : {outputs_dir.resolve()}")
    print(f"[PLOTSET] per_case={do_per_case} comparisons={do_comparisons} include_comm_per_case={include_comm_per_case}")
    print("")

    cases_info: list[tuple[str, Path, float]] = []
    for item in case_items:
        case_path = resolve_case_path(cases_base_dir, item)
        if not case_path.exists():
            raise FileNotFoundError(f"Case YAML not found: {case_path}")

        case_name, dt_hours = get_case_name_and_dt(case_path)

        if enabled_map.get(case_name, True) is False:
            print(f"[SKIP] {case_name} ({case_path.name})")
            continue

        case_out_dir = outputs_dir / case_name
        cases_info.append((case_name, case_out_dir, dt_hours))

    if not cases_info:
        print("[WARN] No enabled cases to plot.")
        return

    if do_per_case:
        for case_name, case_out_dir, dt_hours in cases_info:
            print(f"[PER-CASE] {case_name} -> {case_out_dir}")
            if not case_out_dir.exists():
                print(f"[WARN] outputs folder not found: {case_out_dir} (run the case first)")
                continue
            make_per_case_plots(case_name, case_out_dir, dt_hours=dt_hours, include_community=include_comm_per_case)
            print("")

    if do_comparisons:
        make_comparisons(plotset_name, outputs_dir, cases_info)

    print("[DONE] Plotset finished.")


if __name__ == "__main__":
    make_plots_all(ROOT / "cases" / "plotset.yaml")

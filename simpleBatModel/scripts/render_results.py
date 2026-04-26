from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import argparse
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
from batEnv.utils.community_metrics import (
    COMMUNITY_ID,
    aggregate_community_timeseries,
    compute_community_extra_metrics,
)


def load_plotset_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML must parse to a dict.")
    cfg["_plotset_path"] = str(path.resolve())
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge (override wins)."""
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_from(yaml_path: Path, maybe_rel: str | Path) -> Path:
    p = Path(maybe_rel)
    if p.is_absolute():
        return p.resolve()

    cand = (yaml_path.parent / p).resolve()
    if cand.exists():
        return cand

    return (ROOT / p).resolve()


def _auto_pick_outputs_dir(outputs_dir: Path, plotset_name: str, case_names: list[str]) -> Path:
    """
    Auto-detect where results actually live:
      - outputs_dir/<case_name>
      - outputs_dir/<plotset_name>/<case_name>
    Picks the candidate with the most matching case folders / CSVs.
    """
    candidates = [outputs_dir]

    nested = outputs_dir / plotset_name
    if nested != outputs_dir:
        candidates.append(nested)

    def _score(base: Path) -> tuple[int, int, int]:
        csv_hits = 0
        dir_hits = 0
        for cn in case_names:
            d = base / cn
            if d.is_dir():
                dir_hits += 1
                if any(d.glob("results_house_*.csv")):
                    csv_hits += 1
        return (csv_hits, dir_hits, int(base.is_dir()))

    best = max(candidates, key=_score)

    if best != outputs_dir:
        print(f"[AUTO] outputs_dir adjusted: {outputs_dir.resolve()} -> {best.resolve()}")

    return best


def _is_parent_plotset(cfg: dict) -> bool:
    """
    Parent plotset format:
      plotset: [plotset1.yaml, plotset2.yaml, ...]
    """
    return isinstance(cfg.get("plotset", None), list)


def _is_single_experiment(cfg: dict) -> bool:
    """
    Single experiment YAML format:
      experiment: name
      case_yaml: path/to/case.yaml
    """
    return ("experiment" in cfg) and isinstance(cfg.get("case_yaml", None), (str, Path))


def resolve_case_path(cases_base_dir: Path, item: str) -> Path:
    p = Path(item)
    if not p.is_absolute():
        p = (cases_base_dir / p).resolve()
    return p


def get_case_name(case_yaml_path: Path) -> str:
    c = load_case_yaml(case_yaml_path)
    return str(c.get("case", case_yaml_path.stem))


def _read_meta(case_out_dir: Path) -> dict:
    meta_path = case_out_dir / "meta.yaml"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            m = yaml.safe_load(f) or {}
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def get_dt_hours(case_yaml_path: Path, case_out_dir: Path, dt_default: float = 1.0) -> float:
    meta = _read_meta(case_out_dir)
    if "dt_hours" in meta and meta["dt_hours"] is not None:
        try:
            return float(meta["dt_hours"])
        except Exception:
            pass
    c = load_case_yaml(case_yaml_path)
    time_cfg = c.get("time", {}) if isinstance(c.get("time", {}), dict) else {}
    try:
        return float(time_cfg.get("dt_hours", dt_default))
    except Exception:
        return float(dt_default)


def read_house_csvs(case_out_dir: Path) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(case_out_dir.glob("results_house_*.csv")):
        house_id = csv_path.stem.replace("results_house_", "")
        dfs[house_id] = pd.read_csv(csv_path)
    return dfs


def make_per_case_plots(
    case_name: str,
    case_out_dir: Path,
    dt_hours: float,
    *,
    per_case_root: Path,
    include_community: bool = False,
    case_yaml_path: Optional[Path] = None,
) -> None:
    plots_dir = per_case_root / "houses"
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
            comm_csv = per_case_root / "community_timeseries.csv"
            df_comm.to_csv(comm_csv, index=False)

    case_cfg: dict = {}
    if case_yaml_path is not None:
        case_cfg = load_case_yaml(case_yaml_path)

    for house_id, df in house_dfs.items():
        out_png = plots_dir / f"plot_house_{house_id}.png"

        # If this is the community “house”, it typically has no battery config in YAML
        bat = (case_cfg.get("houses", {}).get(house_id, {}) or {}).get("battery", {}) or {}
        E_min = bat.get("E_min", None)
        E_max = bat.get("E_max", None)

        plot_house_per_case(
            df,
            out_png,
            title=f"{case_name} | {house_id}",
            dt_hours=dt_hours,
            E_min=float(E_min) if E_min is not None else None,
            E_max=float(E_max) if E_max is not None else None,
        )

    print(f"[OK] per-case saved: {per_case_root}")


def make_comparisons(
    plotset_name: str,
    out_root: Path,
    cases_info: list[tuple[str, Path, float]],
) -> None:
    comp_root = out_root / "comparisons"
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
        "P_share",
        "P_share_in",
        "P_share_out",
        "cost_total_step",
        "cost_total_cum",
    ]
    ts_root = comp_root / "timeseries"
    ts_root.mkdir(parents=True, exist_ok=True)

    for house_id in houses_union:
        dfs_for_house = {}
        for case_name, _, _ in cases_info:
            df = case_house_dfs.get(case_name, {}).get(house_id)
            if df is not None:
                dfs_for_house[case_name] = df

        # needs at least 2 cases to compare
        if len(dfs_for_house) < 2:
            continue

        for var in compare_vars:
            out_png = ts_root / f"compare_{var}_house_{house_id}.png"
            plot_compare_timeseries(
                dfs_for_house,
                variable=var,
                outpath=out_png,
                title=f"{plotset_name} | {var} | {house_id}",
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
            "P_imp_max_kW",
            "P_net_grid_max_kW",
            "E_curt_kWh",
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
                plot_compare_metrics(
                    sub,
                    metric=metric,
                    outpath=out_png,
                    title=f"{plotset_name} | {metric} | {house_id}",
                )

        print(f"[OK] comparisons: metric bars saved to {bar_root}")

    print(f"[OK] comparisons root: {comp_root}")


def _run_single_plotset(ps: dict, *, plotset_yaml_path: Path) -> None:
    plotset_name = str(ps.get("plotset", plotset_yaml_path.stem))

    defaults = ps.get("defaults", {}) if isinstance(ps.get("defaults", {}), dict) else {}

    cases_base_dir_val = ps.get("cases_base_dir", defaults.get("cases_base_dir", "cases"))
    cases_base_dir = _resolve_from(plotset_yaml_path, cases_base_dir_val)

    outputs_dir_val = ps.get("outputs_dir", defaults.get("outputs_dir", "results"))
    outputs_dir = _resolve_from(plotset_yaml_path, outputs_dir_val)

    enabled_map = ps.get("enabled", {}) or {}
    if not isinstance(enabled_map, dict):
        raise ValueError("enabled must be a dict mapping case_name -> true/false")

    plots_cfg_parent = defaults.get("plots", {}) if isinstance(defaults.get("plots", {}), dict) else {}
    plots_cfg_child = ps.get("plots", {}) if isinstance(ps.get("plots", {}), dict) else {}
    plots_cfg = _deep_merge(plots_cfg_parent, plots_cfg_child)

    do_per_case = bool(plots_cfg.get("per_case", True))
    do_comparisons = bool(plots_cfg.get("comparisons", True))
    include_comm_per_case = bool(plots_cfg.get("include_community_per_case", True))

    case_items = ps.get("cases", [])
    if not isinstance(case_items, list) or not case_items:
        raise ValueError("plotset.cases must be a non-empty list of case YAML files")

    # Resolve case YAMLs first (needed for auto-detecting outputs_dir)
    resolved_cases: list[tuple[str, Path]] = []
    for item in case_items:
        case_path = resolve_case_path(cases_base_dir, item)
        if not case_path.exists():
            raise FileNotFoundError(f"Case YAML not found: {case_path}")
        case_name = get_case_name(case_path)
        resolved_cases.append((case_name, case_path))

    enabled_case_names = [cn for cn, _ in resolved_cases if enabled_map.get(cn, True) is not False]

    # AUTO: supports both:
    #   outputs_dir/<case_name>
    #   outputs_dir/<plotset_name>/<case_name>
    outputs_dir = _auto_pick_outputs_dir(outputs_dir, plotset_name, enabled_case_names)

    out_root = outputs_dir / "_plots" / plotset_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[PLOTSET] {plotset_name}")
    print(f"[PLOTSET] plotset_yaml: {plotset_yaml_path.resolve()}")
    print(f"[PLOTSET] cases_base_dir: {cases_base_dir.resolve()}")
    print(f"[PLOTSET] outputs_dir  : {outputs_dir.resolve()}")
    print(f"[PLOTSET] out_root     : {out_root.resolve()}")
    print("")

    cases_info: list[tuple[str, Path, float]] = []
    for case_name, case_path in resolved_cases:
        if enabled_map.get(case_name, True) is False:
            print(f"[SKIP] {case_name} ({case_path.name})")
            continue

        case_out_dir = outputs_dir / case_name
        dt_hours = get_dt_hours(case_path, case_out_dir, dt_default=1.0)
        cases_info.append((case_name, case_out_dir, float(dt_hours)))

    if not cases_info:
        print("[WARN] No enabled cases to plot.")
        return

    case_path_by_name = {cn: cp for cn, cp in resolved_cases}

    if do_per_case:
        for case_name, case_out_dir, dt_hours in cases_info:
            if not case_out_dir.exists():
                print(f"[WARN] outputs folder not found: {case_out_dir} (run the case first)")
                continue

            # Per-case plots go inside the same case folder as the CSVs
            per_case_root = case_out_dir / "_plots" / plotset_name

            make_per_case_plots(
                case_name,
                case_out_dir,
                dt_hours=dt_hours,
                per_case_root=per_case_root,
                include_community=include_comm_per_case,
                case_yaml_path=case_path_by_name[case_name],
            )

    if do_comparisons:
        make_comparisons(plotset_name, out_root, cases_info)

    print(f"[DONE] Plotset finished. Output in: {out_root}")
    print("")


def _run_parent_plotset(parent: dict, *, parent_yaml_path: Path) -> None:
    children = parent.get("plotset", [])
    if not isinstance(children, list) or not children:
        raise ValueError("Parent plotset must have a non-empty list under key 'plotset'.")

    enabled_map = parent.get("enabled", {}) or {}
    if not isinstance(enabled_map, dict):
        raise ValueError("Parent enabled must be a dict mapping plotset_name -> true/false")

    parent_defaults = parent.get("defaults", {}) if isinstance(parent.get("defaults", {}), dict) else {}

    print(f"[PARENT PLOTSET] {parent_yaml_path.name}")
    print(f"[PARENT PLOTSET] Found {len(children)} child plotsets")
    print("")

    for child_rel in children:
        child_path = _resolve_from(parent_yaml_path, child_rel)
        if not child_path.exists():
            raise FileNotFoundError(f"Child plotset YAML not found: {child_path}")

        child_cfg = load_plotset_yaml(child_path)
        if _is_parent_plotset(child_cfg):
            _run_parent_plotset(child_cfg, parent_yaml_path=child_path)
            continue

        child_name = str(child_cfg.get("plotset", child_path.stem))
        if enabled_map.get(child_name, True) is False:
            print(f"[SKIP PLOTSET] {child_name} ({child_path.name})")
            print("")
            continue

        child_defaults = child_cfg.get("defaults", {}) if isinstance(child_cfg.get("defaults", {}), dict) else {}
        child_cfg["defaults"] = _deep_merge(parent_defaults, child_defaults)

        _run_single_plotset(child_cfg, plotset_yaml_path=child_path)


def _run_single_experiment_plot(exp: dict, *, exp_yaml_path: Path) -> None:
    """
    Uses the SAME YAML you run for the simulation:
      experiment: name
      enabled: true
      case_yaml: <path>
      defaults:
        outputs_dir: results
        plots: { per_case: true, comparisons: false, include_community_per_case: true }
    """
    name = str(exp.get("experiment", exp_yaml_path.stem))

    if exp.get("enabled", True) is False:
        print(f"[SKIP EXP PLOT] {name} ({exp_yaml_path.name})")
        return

    defaults = exp.get("defaults", {}) if isinstance(exp.get("defaults", {}), dict) else {}

    outputs_dir = Path(defaults.get("outputs_dir", "results"))
    if not outputs_dir.is_absolute():
        outputs_dir = (ROOT / outputs_dir).resolve()

    plots_cfg = defaults.get("plots", {}) if isinstance(defaults.get("plots", {}), dict) else {}
    do_per_case = bool(plots_cfg.get("per_case", True))
    do_comparisons = bool(plots_cfg.get("comparisons", False))  # default false for single
    include_comm = bool(plots_cfg.get("include_community_per_case", True))

    case_yaml_rel = exp.get("case_yaml")
    case_path = _resolve_from(exp_yaml_path, case_yaml_rel)
    if not case_path.exists():
        raise FileNotFoundError(f"Experiment case YAML not found: {case_path}")

    case_name = get_case_name(case_path)
    case_out_dir = outputs_dir / case_name
    dt_hours = get_dt_hours(case_path, case_out_dir, dt_default=1.0)

    out_root = outputs_dir / "_plots" / name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[EXPERIMENT PLOT] {name}")
    print(f"[EXPERIMENT PLOT] case: {case_name} ({case_path})")
    print(f"[EXPERIMENT PLOT] outputs_dir: {outputs_dir}")
    print(f"[EXPERIMENT PLOT] out_root   : {out_root}")
    print("")

    if not case_out_dir.exists():
        print(f"[WARN] outputs folder not found: {case_out_dir} (run the experiment first)")
        return

    if do_per_case:
        per_case_root = case_out_dir / "_plots" / name
        make_per_case_plots(
            case_name,
            case_out_dir,
            dt_hours=dt_hours,
            per_case_root=per_case_root,
            include_community=include_comm,
            case_yaml_path=case_path,  # IMPORTANT: pass yaml path for SOC bounds
        )

    if do_comparisons:
        # comparisons won't do much with one case, but metrics export can still be useful
        make_comparisons(name, out_root, [(case_name, case_out_dir, float(dt_hours))])

    print(f"[DONE] Single experiment plotting finished. Output in: {out_root}")
    print("")


def make_plots_all(plotset_yaml: str | Path) -> None:
    plotset_yaml_path = Path(plotset_yaml)
    if not plotset_yaml_path.is_absolute():
        plotset_yaml_path = (ROOT / plotset_yaml_path).resolve()

    cfg = load_plotset_yaml(plotset_yaml_path)

    # allow plotting using the SAME experiment.yaml
    if _is_single_experiment(cfg):
        _run_single_experiment_plot(cfg, exp_yaml_path=plotset_yaml_path)
        return

    if _is_parent_plotset(cfg):
        _run_parent_plotset(cfg, parent_yaml_path=plotset_yaml_path)
    else:
        _run_single_plotset(cfg, plotset_yaml_path=plotset_yaml_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plotset",
        default=str(ROOT / "cases" / "plotset_parent.yaml"),
        help="Path to parent plotset YAML (or experiment.yaml for single-case plotting).",
    )
    args = ap.parse_args()
    make_plots_all(args.plotset)


if __name__ == "__main__":
    main()
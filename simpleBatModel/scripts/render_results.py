from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import (
    add_src_to_path,
    collect_case_files,
    deep_merge,
    is_single_experiment,
    load_yaml_dict,
    read_house_csvs,
    resolve_from,
    setup_logging,
)

add_src_to_path(ROOT)

from batEnv.io import load_case_yaml, validate_plotset_cfg  # noqa: E402
from batEnv.plotting import (  # noqa: E402
    compute_summary_metrics,
    plot_compare_metrics,
    plot_compare_timeseries,
    plot_house_per_case,
    plot_summary_dashboard,
)
from batEnv.utils.community_metrics import (  # noqa: E402
    COMMUNITY_ID,
    aggregate_community_timeseries,
    compute_community_extra_metrics,
    compute_fairness_metrics,
)


logger = logging.getLogger(__name__)


# ---------- plotset YAML ----------

def load_plotset_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    cfg = load_yaml_dict(p)
    validate_plotset_cfg(cfg)
    cfg["_plotset_path"] = str(p.resolve())
    return cfg


def _is_parent_plotset(cfg: dict) -> bool:
    """Parent plotset format: `plotset: [child1.yaml, ...]` (list, not str)."""
    return isinstance(cfg.get("plotset", None), list)


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
    except (OSError, yaml.YAMLError):
        return {}
    return m if isinstance(m, dict) else {}


def get_dt_hours(case_yaml_path: Path, case_out_dir: Path, dt_default: float = 1.0) -> float:
    """Prefer dt_hours from results meta.yaml; fall back to case YAML, then default."""
    meta = _read_meta(case_out_dir)
    if "dt_hours" in meta and meta["dt_hours"] is not None:
        try:
            return float(meta["dt_hours"])
        except (TypeError, ValueError):
            pass
    c = load_case_yaml(case_yaml_path)
    time_cfg = c.get("time", {}) if isinstance(c.get("time", {}), dict) else {}
    try:
        return float(time_cfg.get("dt_hours", dt_default))
    except (TypeError, ValueError):
        return float(dt_default)


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
        logger.info("Auto-adjusted outputs_dir: %s -> %s", outputs_dir.resolve(), best.resolve())
    return best


# ---------- per-case plots ----------

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
        logger.warning("%s: no results_house_*.csv in %s", case_name, case_out_dir)
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

        # Community "house" typically has no battery config in YAML
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

    logger.info("Per-case plots saved: %s", per_case_root)


# ---------- comparisons across cases ----------

_COMPARE_VARS = (
    "E", "P_imp", "P_exp", "P_ch", "P_dis",
    "cost_step", "cost_cum", "P_net_grid", "P_simul_imp_exp",
)
_PREFERRED_BAR_METRICS = (
    "Cost_total_EUR", "E_imp_kWh", "E_exp_kWh", "E_ch_kWh", "E_dis_kWh",
    "E_end_kWh", "E_simul_imp_exp_kWh", "P_imp_max_kW", "P_net_grid_max_kW",
    "E_curt_kWh",
)


def make_comparisons(
    plotset_name: str,
    out_root: Path,
    cases_info: list[tuple[str, Path, float]],
) -> None:
    comp_root = out_root / "comparisons"
    comp_root.mkdir(parents=True, exist_ok=True)

    case_house_dfs: dict[str, dict[str, pd.DataFrame]] = {}
    houses_union: set[str] = set()
    dt_by_case: dict[str, float] = {}

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

    metrics_rows = []
    for case_name, case_out_dir, dt_hours in cases_info:
        if not case_out_dir.exists():
            continue
        hd = case_house_dfs.get(case_name, {})

        # Per-house metrics for this case
        case_house_rows = []
        for house_id, df in hd.items():
            m = compute_summary_metrics(df, dt_hours=dt_hours)
            if house_id == COMMUNITY_ID:
                m.update(compute_community_extra_metrics(df, dt_hours=dt_hours))
            m["case"] = case_name
            m["house"] = house_id
            metrics_rows.append(m)
            case_house_rows.append(m)

        # Fairness across the (real) houses for this case — attach to the
        # community row so it shows up under house=_COMMUNITY in the dashboard.
        if case_house_rows:
            df_case = pd.DataFrame(case_house_rows)
            fairness = compute_fairness_metrics(df_case, metric="Cost_total_EUR")
            if fairness:
                # find the community row in metrics_rows we just appended for this case
                for row in metrics_rows:
                    if row["case"] == case_name and row["house"] == COMMUNITY_ID:
                        row.update(fairness)
                        break

    if metrics_rows:
        metrics_all = pd.DataFrame(metrics_rows).set_index(["case", "house"]).sort_index()
        metrics_csv = comp_root / "metrics_all.csv"
        metrics_all.to_csv(metrics_csv)
        logger.info("Metrics table: %s", metrics_csv)

        # One-page summary dashboard at community level (SS/SC/Cost/Fairness)
        if any(metrics_all.index.get_level_values("house") == COMMUNITY_ID):
            dash_path = comp_root / "summary_dashboard.png"
            plot_summary_dashboard(
                metrics_all, dash_path,
                title=f"{plotset_name} — community summary (sorted by total cost)",
                sort_by="Cost_total_EUR",
            )
            logger.info("Summary dashboard: %s", dash_path)
    else:
        metrics_all = pd.DataFrame()
        logger.warning("No metrics could be computed (no CSVs found).")

    # Per-house timeseries comparisons (need >=2 cases per house to be meaningful)
    ts_root = comp_root / "timeseries"
    ts_root.mkdir(parents=True, exist_ok=True)
    for house_id in houses_union:
        dfs_for_house = {
            cn: case_house_dfs.get(cn, {}).get(house_id)
            for cn, _, _ in cases_info
        }
        dfs_for_house = {k: v for k, v in dfs_for_house.items() if v is not None}
        if len(dfs_for_house) < 2:
            continue

        for var in _COMPARE_VARS:
            out_png = ts_root / f"compare_{var}_house_{house_id}.png"
            plot_compare_timeseries(
                dfs_for_house,
                variable=var,
                outpath=out_png,
                title=f"{plotset_name} | {var} | {house_id}",
                dt_hours_by_case=dt_by_case,
            )
        logger.info("Comparisons: timeseries for %s", house_id)

    # Per-house metric bars
    if not metrics_all.empty:
        bar_root = comp_root / "metrics"
        bar_root.mkdir(parents=True, exist_ok=True)

        metric_list = [m for m in _PREFERRED_BAR_METRICS if m in metrics_all.columns]

        for house_id in houses_union:
            try:
                sub = metrics_all.xs(house_id, level="house")
            except KeyError:
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

        logger.info("Comparisons: metric bars saved to %s", bar_root)

    logger.info("Comparisons root: %s", comp_root)


# ---------- single plotset ----------

def _run_single_plotset(ps: dict, *, plotset_yaml_path: Path) -> None:
    plotset_name = str(ps.get("plotset", plotset_yaml_path.stem))
    defaults = ps.get("defaults", {}) if isinstance(ps.get("defaults", {}), dict) else {}

    cases_base_dir_val = ps.get("cases_base_dir", defaults.get("cases_base_dir", "cases"))
    cases_base_dir = resolve_from(plotset_yaml_path, cases_base_dir_val, root=ROOT)

    outputs_dir_val = ps.get("outputs_dir", defaults.get("outputs_dir", "results"))
    outputs_dir = resolve_from(plotset_yaml_path, outputs_dir_val, root=ROOT)

    enabled_map = ps.get("enabled", {}) or {}
    if not isinstance(enabled_map, dict):
        raise ValueError("enabled must be a dict mapping case_name -> true/false")

    plots_cfg_parent = defaults.get("plots", {}) if isinstance(defaults.get("plots", {}), dict) else {}
    plots_cfg_child = ps.get("plots", {}) if isinstance(ps.get("plots", {}), dict) else {}
    plots_cfg = deep_merge(plots_cfg_parent, plots_cfg_child)

    do_per_case = bool(plots_cfg.get("per_case", True))
    do_comparisons = bool(plots_cfg.get("comparisons", True))
    include_comm_per_case = bool(plots_cfg.get("include_community_per_case", True))

    case_files = collect_case_files(ps, cases_base_dir)

    # Resolve case YAMLs first (needed for auto-detecting outputs_dir)
    resolved_cases: list[tuple[str, Path]] = []
    for case_path in case_files:
        if not case_path.exists():
            raise FileNotFoundError(f"Case YAML not found: {case_path}")
        case_name = get_case_name(case_path)
        resolved_cases.append((case_name, case_path))

    enabled_case_names = [cn for cn, _ in resolved_cases if enabled_map.get(cn, True) is not False]
    outputs_dir = _auto_pick_outputs_dir(outputs_dir, plotset_name, enabled_case_names)

    out_root = outputs_dir / "_plots" / plotset_name
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Running plotset: %s", plotset_name)
    logger.info("  plotset_yaml  : %s", plotset_yaml_path.resolve())
    logger.info("  cases_base_dir: %s", cases_base_dir.resolve())
    logger.info("  outputs_dir   : %s", outputs_dir.resolve())
    logger.info("  out_root      : %s", out_root.resolve())

    cases_info: list[tuple[str, Path, float]] = []
    for case_name, case_path in resolved_cases:
        if enabled_map.get(case_name, True) is False:
            logger.info("Skipping case: %s (%s)", case_name, case_path.name)
            continue
        case_out_dir = outputs_dir / case_name
        dt_hours = get_dt_hours(case_path, case_out_dir, dt_default=1.0)
        cases_info.append((case_name, case_out_dir, float(dt_hours)))

    if not cases_info:
        logger.warning("No enabled cases to plot.")
        return

    case_path_by_name = {cn: cp for cn, cp in resolved_cases}

    if do_per_case:
        for case_name, case_out_dir, dt_hours in cases_info:
            if not case_out_dir.exists():
                logger.warning("Outputs folder not found: %s (run the case first)", case_out_dir)
                continue
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

    logger.info("Plotset finished. Output in: %s", out_root)


# ---------- parent plotset ----------

def _run_parent_plotset(parent: dict, *, parent_yaml_path: Path) -> None:
    children = parent.get("plotset", [])
    if not isinstance(children, list) or not children:
        raise ValueError("Parent plotset must have a non-empty list under key 'plotset'.")

    enabled_map = parent.get("enabled", {}) or {}
    if not isinstance(enabled_map, dict):
        raise ValueError("Parent enabled must be a dict mapping plotset_name -> true/false")

    parent_defaults = parent.get("defaults", {}) if isinstance(parent.get("defaults", {}), dict) else {}

    logger.info("Running parent plotset: %s", parent_yaml_path.name)
    logger.info("  found %d child plotsets", len(children))

    for child_rel in children:
        child_path = resolve_from(parent_yaml_path, child_rel, root=ROOT)
        if not child_path.exists():
            raise FileNotFoundError(f"Child plotset YAML not found: {child_path}")

        child_cfg = load_plotset_yaml(child_path)
        if _is_parent_plotset(child_cfg):
            _run_parent_plotset(child_cfg, parent_yaml_path=child_path)
            continue

        child_name = str(child_cfg.get("plotset", child_path.stem))
        if enabled_map.get(child_name, True) is False:
            logger.info("Skipping child plotset: %s (%s)", child_name, child_path.name)
            continue

        child_defaults = child_cfg.get("defaults", {}) if isinstance(child_cfg.get("defaults", {}), dict) else {}
        child_cfg["defaults"] = deep_merge(parent_defaults, child_defaults)

        _run_single_plotset(child_cfg, plotset_yaml_path=child_path)


# ---------- single experiment plotting ----------

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
        logger.info("Skipping experiment plotting: %s (%s)", name, exp_yaml_path.name)
        return

    defaults = exp.get("defaults", {}) if isinstance(exp.get("defaults", {}), dict) else {}

    outputs_dir = Path(defaults.get("outputs_dir", "results"))
    if not outputs_dir.is_absolute():
        outputs_dir = (ROOT / outputs_dir).resolve()

    plots_cfg = defaults.get("plots", {}) if isinstance(defaults.get("plots", {}), dict) else {}
    do_per_case = bool(plots_cfg.get("per_case", True))
    do_comparisons = bool(plots_cfg.get("comparisons", False))
    include_comm = bool(plots_cfg.get("include_community_per_case", True))

    case_yaml_rel = exp.get("case_yaml")
    case_path = resolve_from(exp_yaml_path, case_yaml_rel, root=ROOT)
    if not case_path.exists():
        raise FileNotFoundError(f"Experiment case YAML not found: {case_path}")

    case_name = get_case_name(case_path)
    case_out_dir = outputs_dir / case_name
    dt_hours = get_dt_hours(case_path, case_out_dir, dt_default=1.0)

    out_root = outputs_dir / "_plots" / name
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting experiment: %s", name)
    logger.info("  case       : %s (%s)", case_name, case_path)
    logger.info("  outputs_dir: %s", outputs_dir)
    logger.info("  out_root   : %s", out_root)

    if not case_out_dir.exists():
        logger.warning("Outputs folder not found: %s (run the experiment first)", case_out_dir)
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
        # Comparisons with one case won't draw timeseries lines, but the metrics
        # CSV is still useful.
        make_comparisons(name, out_root, [(case_name, case_out_dir, float(dt_hours))])

    logger.info("Single experiment plotting finished. Output in: %s", out_root)


# ---------- entry point ----------

def make_plots_all(plotset_yaml: str | Path) -> None:
    plotset_yaml_path = Path(plotset_yaml)
    if not plotset_yaml_path.is_absolute():
        plotset_yaml_path = (ROOT / plotset_yaml_path).resolve()

    cfg = load_plotset_yaml(plotset_yaml_path)

    if is_single_experiment(cfg):
        _run_single_experiment_plot(cfg, exp_yaml_path=plotset_yaml_path)
        return

    if _is_parent_plotset(cfg):
        _run_parent_plotset(cfg, parent_yaml_path=plotset_yaml_path)
    else:
        _run_single_plotset(cfg, plotset_yaml_path=plotset_yaml_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plotset",
        default=str(ROOT / "cases" / "plotset_parent.yaml"),
        help="Path to parent plotset YAML (or experiment.yaml for single-case plotting).",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    args = ap.parse_args()
    setup_logging(verbose=args.verbose)
    make_plots_all(args.plotset)


if __name__ == "__main__":
    main()

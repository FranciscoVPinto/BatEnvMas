from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import (
    add_src_to_path,
    load_yaml_dict,
    read_house_csvs,
    setup_logging,
)

add_src_to_path(ROOT)

from batEnv.io import load_case_yaml, validate_runset_cfg  # noqa: E402
from batEnv.plotting import compute_summary_metrics  # noqa: E402
from batEnv.utils.community_metrics import (  # noqa: E402
    COMMUNITY_ID,
    aggregate_community_timeseries,
    compute_community_extra_metrics,
)


logger = logging.getLogger(__name__)


def load_runset(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    cfg = load_yaml_dict(p)
    validate_runset_cfg(cfg)
    cfg["_runset_path"] = str(p.resolve())
    return cfg


def summarize_runset(runset_yaml: str | Path) -> Path:
    rs = load_runset(runset_yaml)
    runset_name = str(rs.get("runset", Path(runset_yaml).stem))

    base_dir = Path(rs.get("cases_base_dir", "cases"))
    if not base_dir.is_absolute():
        base_dir = (ROOT / base_dir).resolve()

    defaults = rs.get("defaults", {}) if isinstance(rs.get("defaults", {}), dict) else {}
    outputs_dir = Path(defaults.get("outputs_dir", "results"))
    if not outputs_dir.is_absolute():
        outputs_dir = (ROOT / outputs_dir).resolve()

    enabled_map = rs.get("enabled", {}) or {}
    case_files = rs.get("cases", []) or []
    if not isinstance(case_files, list) or not case_files:
        raise ValueError("runset.cases must be a non-empty list")

    rows = []
    for rel in case_files:
        case_path = Path(rel)
        if not case_path.is_absolute():
            case_path = (base_dir / case_path).resolve()
        if not case_path.exists():
            logger.warning("Missing case yaml: %s", case_path)
            continue

        cfg = load_case_yaml(case_path)
        case_name = str(cfg.get("case", case_path.stem))
        if enabled_map.get(case_name, True) is False:
            continue

        time_cfg = cfg.get("time", {}) if isinstance(cfg.get("time", {}), dict) else {}
        dt_hours = float(time_cfg.get("dt_hours", 1.0))

        out_dir = outputs_dir / case_name
        if not out_dir.exists():
            logger.warning("Missing outputs for case '%s': %s", case_name, out_dir)
            continue

        house_dfs = read_house_csvs(out_dir)
        for hid, df in house_dfs.items():
            m = compute_summary_metrics(df, dt_hours=dt_hours)
            m["case"] = case_name
            m["house"] = hid
            rows.append(m)

        df_comm = aggregate_community_timeseries(house_dfs)
        if not df_comm.empty:
            m = compute_summary_metrics(df_comm, dt_hours=dt_hours)
            m.update(compute_community_extra_metrics(df_comm, dt_hours=dt_hours))
            m["case"] = case_name
            m["house"] = COMMUNITY_ID
            rows.append(m)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No results found to summarize.")

    summary = summary.set_index(["case", "house"]).sort_index()

    outdir = outputs_dir / "_summaries"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{runset_name}_summary.csv"
    summary.to_csv(outpath)
    logger.info("Wrote summary: %s", outpath)
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runset", default=str(ROOT / "cases" / "runset.yaml"), help="Path to runset YAML")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    args = ap.parse_args()
    setup_logging(verbose=args.verbose)
    summarize_runset(args.runset)


if __name__ == "__main__":
    main()

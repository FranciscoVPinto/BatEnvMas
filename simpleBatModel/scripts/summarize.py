from __future__ import annotations

import sys
from pathlib import Path
import argparse
import yaml
import pandas as pd

# ---- Spyder-proof: add src/ to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batEnv.io import load_case_yaml
from batEnv.plotting import compute_summary_metrics
from batEnv.utils.community_metrics import COMMUNITY_ID, aggregate_community_timeseries, compute_community_extra_metrics


def load_runset(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("runset YAML must parse to a dict.")
    cfg["_runset_path"] = str(path.resolve())
    return cfg


def read_house_csvs(case_out_dir: Path) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(case_out_dir.glob("results_house_*.csv")):
        hid = csv_path.stem.replace("results_house_", "")
        dfs[hid] = pd.read_csv(csv_path)
    return dfs


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
            print(f"[WARN] Missing case yaml: {case_path}")
            continue

        cfg = load_case_yaml(case_path)
        case_name = str(cfg.get("case", case_path.stem))
        if enabled_map.get(case_name, True) is False:
            continue

        time_cfg = cfg.get("time", {}) if isinstance(cfg.get("time", {}), dict) else {}
        dt_hours = float(time_cfg.get("dt_hours", 1.0))

        out_dir = outputs_dir / case_name
        if not out_dir.exists():
            print(f"[WARN] Missing outputs for case '{case_name}': {out_dir}")
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
    print(f"[OK] Wrote summary: {outpath}")
    return outpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runset", default=str(ROOT / "cases" / "runset.yaml"), help="Path to runset YAML")
    args = ap.parse_args()
    summarize_runset(args.runset)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml

from batEnv.plotting.simple import make_house_plots


def _load_meta(case_dir: Path) -> dict:
    p = case_dir / "meta.yaml"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        m = yaml.safe_load(f) or {}
    return m if isinstance(m, dict) else {}


def make_plots(case_output_dir: str):
    case_dir = Path(case_output_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case output dir not found: {case_dir}")

    meta = _load_meta(case_dir)

    plots_dir = case_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(case_dir.glob("results_house_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results_house_*.csv found in {case_dir}")

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        house_id = csv_path.stem.replace("results_house_", "")
        out_house_dir = plots_dir / f"house_{house_id}"
        make_house_plots(df, out_house_dir, house_id=house_id, meta=meta, title_prefix="")
        print(f"[OK] Plots created: {out_house_dir}")

    print(f"[OK] All plots saved to: {plots_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_output_dir", help="Path to results/<case_name>")
    args = ap.parse_args()
    make_plots(args.case_output_dir)


if __name__ == "__main__":
    main()

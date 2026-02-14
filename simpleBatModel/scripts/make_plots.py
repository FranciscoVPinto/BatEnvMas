from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from batEnv.plotting import make_plot_simple


def make_plots(case_output_dir: str):
    case_dir = Path(case_output_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case output dir not found: {case_dir}")

    plots_dir = case_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(case_dir.glob("results_house_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results_house_*.csv found in {case_dir}")

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        house_id = csv_path.stem.replace("results_house_", "")
        out_png = plots_dir / f"plot_house_{house_id}.png"
        make_plot_simple(df, out_png)
        print(f"[OK] Plot created: {out_png}")

    print(f"[OK] All plots saved to: {plots_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_output_dir", help="Path to outputs/<case_name> (e.g., outputs/case_1house)")
    args = ap.parse_args()
    make_plots(args.case_output_dir)


if __name__ == "__main__":
    main()

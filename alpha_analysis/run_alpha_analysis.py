from __future__ import annotations
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd

# Project imports
from env.community import Community
from env.state import StateVars
from utils.dataprocessor import YAMLParser
from utils.utilities import ConfigsParser

# Analysis modules
from analysis.alphas import (
    build_alpha_configs_pv_global, normalize_alpha_cols,
    alpha_fixed_from_shares, alpha_fixed_from_df,
    alpha_hierarchical,
)
from analysis.allocate import allocate_pv_global_to_load, build_receiver_mask
from analysis.metrics import eval_metrics, metrics_table, fairness_index, jains_index


def parse_args():
    p = argparse.ArgumentParser(description="Alpha analysis modular runner (clean version)")
    p.add_argument("--horizon", type=int, default=7*24*4, help="H (steps of 15-min)")
    p.add_argument("--apply-eligibility", action="store_true", help="mask out injectors from receiving")
    p.add_argument("--alpha-csv", type=str, default="", help="CSV for time-varying fixed alpha (H rows x n columns)")
    p.add_argument("--exp-yaml", type=str, default="exp_name.yaml", help="experiment YAML in configs/ folder")
    p.add_argument("--save-outputs", action="store_true", help="save CSV outputs")
    p.add_argument("--out-dir", default="report", help="output directory for images/CSVs")
    p.add_argument("--img-format", default="png", choices=["png","pdf","svg"], help="image format")
    p.add_argument("--dpi", type=int, default=144, help="image DPI")
    return p.parse_args()


def main():
    args = parse_args()
    args.full_report = True
    t0 = time.time()

    # === Setup paths and configuration ===
    cwd = Path.cwd()
    datafolder = cwd.parent / "Data"
    configs_folder = cwd.parent / "configs"

    exp_name = YAMLParser().load_yaml(configs_folder / args.exp_yaml)["exp_name"]
    configs = ConfigsParser(configs_folder, exp_name)
    file_ag_conf, file_apps_conf, file_scene_conf, file_prob_conf, file_vars, file_experiment, ppo_config = configs.get_configs()

    # === Build community and extract data ===
    com = Community(file_ag_conf, file_scene_conf, file_prob_conf, datafolder / "dataset_gecad_clean.csv")
    _ = StateVars(file_vars)

    agents = list(com.agents.keys())
    load_df_full = pd.DataFrame({aid: com.agents[aid].data["load"] for aid in agents})
    pv_df_full   = pd.DataFrame({aid: com.agents[aid].data["gen"]  for aid in agents})

    H_eff = min(args.horizon, len(load_df_full), len(pv_df_full))
    load_df = load_df_full.iloc[:H_eff, :].copy()
    pv_df   = pv_df_full.iloc[:H_eff, :].copy().reindex(columns=load_df.columns)

    pv_global = pv_df.sum(axis=1)
    load_global = load_df.sum(axis=1)

    # === Define alphas ===
    alpha_cfgs: dict[str, np.ndarray] = build_alpha_configs_pv_global(load_df)

    # Fixed shares (Art. 29)
    fixed_custom = alpha_fixed_from_shares({"ag1": 0.5, "ag2": 0.3, "ag3": 0.2}, agents, H_eff)
    alpha_cfgs["FixedCustom"] = normalize_alpha_cols(fixed_custom)

    # Hierarchical (Art. 31)
    groups = {"Bloco1": ["ag1", "ag2"], "Bloco2": ["ag3"]}
    alpha_hier = alpha_hierarchical(
        groups, agents, H_eff,
        inner_rule="InstantLoad",
        outer_shares={"Bloco1": 0.6, "Bloco2": 0.4},
        load_df=load_df
    )
    alpha_cfgs["Hierarchical(ILxFix)"] = normalize_alpha_cols(alpha_hier)

    # Optional CSV-based alpha (Art. 30)
    if args.alpha_csv:
        alpha_tv_df = pd.read_csv(args.alpha_csv)
        fixed_timevar = alpha_fixed_from_df(alpha_tv_df, agents, H_eff)
        alpha_cfgs["FixedTimeVar(CSV)"] = normalize_alpha_cols(fixed_timevar)

    # Eligibility mask
    elig_mask = build_receiver_mask(load_df, pv_df) if args.apply_eligibility else None

    # === Run allocations ===
    allocations, exports, imports, pv_used_by_alpha = {}, {}, {}, {}

    for name, alpha in alpha_cfgs.items():
        np.testing.assert_allclose(alpha.sum(axis=0), 1.0, rtol=1e-10, atol=1e-10)
        alloc_df, export_series, import_series = allocate_pv_global_to_load(
            pv_global, load_df, alpha, agents, eligible_mask=elig_mask
        )
        allocations[name] = alloc_df
        exports[name] = export_series
        imports[name] = import_series
        pv_used_by_alpha[name] = alloc_df.sum(axis=1)

    # === Compute metrics (only fairness & alpha comparison) ===
    summary = {}
    for name in allocations:
        m = {}
        m["Fairness"] = fairness_index(allocations[name], load_df)
        m["Jain"] = jains_index(allocations[name], load_df)
        summary[name] = m

    table = pd.DataFrame(summary).T
    print("\n=== Fairness Metrics by Alpha Configuration ===")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    # === Generate full report (optional) ===
    if args.full_report:
        from viz_report import generate_full_report
        timestamps = load_df.index
        alpha_by_cfg_df = {k: pd.DataFrame(v.T, index=timestamps, columns=agents) for k, v in alpha_cfgs.items()}
        metrics_df = table.copy()
        exp_stem = Path(args.exp_yaml).stem
        generate_full_report(
            out_dir=args.out_dir,
            exp_name=exp_stem,
            timestamps=timestamps,
            agents=agents,
            load_df=load_df,
            pv_df=pv_df,
            alloc_by_cfg=allocations,
            metrics_by_cfg=metrics_df,
            alpha_by_cfg=alpha_by_cfg_df,
            receiver_mask=None,
            dpi=args.dpi,
            img_format=args.img_format,
        )

    # === Save optional CSV outputs ===
    if args.save_outputs:
        outdir = cwd / "outputs" / exp_name
        outdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(outdir / "fairness_summary.csv")
        print(f"\nFairness results saved to {outdir}")

    print(f"\nAnalysis completed in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()

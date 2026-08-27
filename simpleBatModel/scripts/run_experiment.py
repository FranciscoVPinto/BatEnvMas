from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
    resolve_from,
    setup_logging,
)

add_src_to_path(ROOT)

import yaml  

from batEnv.io import load_case_yaml, validate_runset_cfg  
from run_case import run_case  


logger = logging.getLogger(__name__)


def load_runset(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    cfg = load_yaml_dict(p)
    validate_runset_cfg(cfg)
    cfg["_runset_path"] = str(p.resolve())
    return cfg


def _case_name_from_yaml(case_yaml_path: Path) -> str:
    try:
        c = load_case_yaml(case_yaml_path)
    except (OSError, ValueError):
        return case_yaml_path.stem
    name = c.get("case", None)
    return name if isinstance(name, str) and name.strip() else case_yaml_path.stem


def _is_parent_runset(cfg: dict) -> bool:
    """Parent runset format: `runset: [child1.yaml, child2.yaml, ...]` (list, not str)."""
    return isinstance(cfg.get("runset", None), list)


def _rebase_outputs(configured: Path, outputs_root: str | None) -> Path:
    """Redirect a runset's `outputs_dir` to a different root, keeping the layout.

    Used by `--outputs-dir`, to re-run a full battery into a clean directory
    WITHOUT overwriting the existing `results/` (which backs the dissertation's
    numbers). The first path component is swapped, so the per-runset structure
    is preserved:

        results/full_horizon_sweep   --outputs-dir results_v2 -->  results_v2/full_horizon_sweep
        results/alfa_deg_sweep       --outputs-dir results_v2 -->  results_v2/alfa_deg_sweep
        results                      --outputs-dir results_v2 -->  results_v2
    """
    if not outputs_root:
        return configured if configured.is_absolute() else (ROOT / configured).resolve()

    new_root = Path(outputs_root)
    if not new_root.is_absolute():
        new_root = (ROOT / new_root).resolve()

    rel = configured if not configured.is_absolute() else Path(configured.name)
    tail = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path()
    return (new_root / tail).resolve()



def _run_single_runset(runset: dict, *, runset_yaml_path: Path, skip_existing: bool = False,
                       outputs_root: str | None = None) -> None:
    runset_name = str(runset.get("runset", runset_yaml_path.stem))

    base_dir = Path(runset.get("cases_base_dir", "cases"))
    if not base_dir.is_absolute():
        base_dir = (ROOT / base_dir).resolve()

    defaults = runset.get("defaults", {}) if isinstance(runset.get("defaults", {}), dict) else {}

    outputs_dir = _rebase_outputs(Path(defaults.get("outputs_dir", "results")), outputs_root)

    tee_default = bool(defaults.get("tee", False))

    solver_defaults = defaults.get("solver", None)
    if solver_defaults is not None and not isinstance(solver_defaults, dict):
        raise ValueError("defaults.solver must be a dict (e.g. {name: 'highs', options: {...}})")
    solver_name = str((solver_defaults or {}).get("name", "highs"))
    solver_opts = (solver_defaults or {}).get("options", None)

    time_defaults = defaults.get("time", None)
    if time_defaults is not None and not isinstance(time_defaults, dict):
        raise ValueError(
            "defaults.time must be a dict (e.g. {horizon: 96, dt_hours: 0.25, start: '2025-01-01 00:00'})"
        )

    enabled_map = runset.get("enabled", {}) or {}
    if not isinstance(enabled_map, dict):
        raise ValueError("enabled must be a dict mapping case_name -> true/false")

    case_files = collect_case_files(runset, base_dir)

    # Sweep: each entry is run for each base case (cartesian product). When
    # absent, behave as a single 'identity' entry — preserves legacy semantics.
    sweep_entries = runset.get("sweep") or []
    if not sweep_entries:
        sweep_entries = [{"suffix": "", "overrides": {}}]

    logger.info("Running runset: %s", runset_name)
    logger.info("  base dir: %s", base_dir.resolve())
    logger.info("  outputs : %s", outputs_dir.resolve())
    if isinstance(time_defaults, dict) and "horizon" in time_defaults:
        logger.info("  horizon : %s", time_defaults.get("horizon"))
    if len(sweep_entries) > 1:
        logger.info("  sweep   : %d entries × %d base cases = %d total runs",
                    len(sweep_entries), len(case_files), len(sweep_entries) * len(case_files))

    sweep_tmp_dir = outputs_dir / "_sweep_tmp"

    for case_path in case_files:
        if not case_path.exists():
            raise FileNotFoundError(f"Case YAML not found: {case_path}")

        base_case_name = _case_name_from_yaml(case_path)
        if enabled_map.get(base_case_name, True) is False:
            logger.info("Skipping case: %s (%s)", base_case_name, case_path.name)
            continue

        for sweep in sweep_entries:
            suffix = str(sweep.get("suffix", "")).strip()
            overrides = sweep.get("overrides") or {}
            sweep_time = sweep.get("time_override") or {}
            sweep_subdir = sweep.get("outputs_subdir")

            # Effective time override: runset defaults overlaid with sweep entry.
            effective_time = dict(time_defaults or {})
            effective_time.update(sweep_time)
            effective_time_override = effective_time or None

            # Per-entry outputs_dir (nested layout when sweep specifies subdir).
            run_outputs_dir = outputs_dir / sweep_subdir if sweep_subdir else outputs_dir

            if not suffix and not overrides and not sweep_time and not sweep_subdir:
                # Identity sweep — run the original case YAML directly.
                run_path = str(case_path)
                run_name = base_case_name
            else:
                # Build a fully-merged temporary YAML so run_case sees a
                # standalone, validated configuration.
                base_cfg = load_case_yaml(case_path)
                merged = deep_merge(base_cfg, overrides)
                run_name = f"{base_case_name}__{suffix}" if suffix else base_case_name
                merged["case"] = run_name

                sweep_tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = sweep_tmp_dir / f"{run_name}.yaml"
                with tmp_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(merged, f, sort_keys=False)
                run_path = str(tmp_path)

            if skip_existing and (run_outputs_dir / run_name / "meta.yaml").exists():
                logger.info("Skipping (already done): %s", run_name)
                continue

            logger.info("Running case: %s (%s)%s",
                        run_name, Path(run_path).name,
                        f" -> {sweep_subdir}/" if sweep_subdir else "")
            try:
                run_case(
                    run_path,
                    outputs_dir=str(run_outputs_dir),
                    tee=tee_default,
                    time_override=effective_time_override,
                    solver=solver_name,
                    solver_options=solver_opts,
                )
            except Exception as case_err:  # noqa: BLE001
                logger.error("Case %s FAILED — skipping. Reason: %s", run_name, case_err)

    logger.info("All enabled cases executed.")



def _run_parent_runset(parent: dict, *, parent_yaml_path: Path, skip_existing: bool = False,
                       outputs_root: str | None = None) -> None:
    """
    Parent runset YAML example:

        runset:
          - runsets/pv/runset_pv_study.yaml
        enabled:
          pv_study: true
        defaults:
          outputs_dir: results
          tee: false
          time: {horizon: 96, dt_hours: 0.25, start: "2025-01-01 00:00"}
    """
    parent_defaults = parent.get("defaults", {}) if isinstance(parent.get("defaults", {}), dict) else {}
    enabled_map = parent.get("enabled", {}) if isinstance(parent.get("enabled", {}), dict) else {}

    runset_list = parent.get("runset", [])
    if not isinstance(runset_list, list) or not runset_list:
        raise ValueError("Parent runset must define a non-empty list under 'runset'.")

    for rel in runset_list:
        runset_yaml_path = resolve_from(parent_yaml_path, rel, root=ROOT)
        if not runset_yaml_path.exists():
            raise FileNotFoundError(f"Child runset YAML not found: {runset_yaml_path}")

        rs = load_runset(runset_yaml_path)
        rs_name = str(rs.get("runset", runset_yaml_path.stem))

        if enabled_map.get(rs_name, True) is False:
            logger.info("Skipping child runset: %s (%s)", rs_name, runset_yaml_path.name)
            continue

        # Merge defaults: parent < child (child wins)
        child_defaults = rs.get("defaults", {}) if isinstance(rs.get("defaults", {}), dict) else {}
        rs["defaults"] = deep_merge(parent_defaults, child_defaults)

        _run_single_runset(rs, runset_yaml_path=runset_yaml_path, skip_existing=skip_existing,
                           outputs_root=outputs_root)



def _run_single_experiment(exp: dict, *, exp_yaml_path: Path, outputs_root: str | None = None) -> None:
    """
    Single-experiment YAML example:

        experiment: name
        enabled: true
        case_yaml: scenarios/pv/study/pv01_equal_export.yaml
        defaults:
          outputs_dir: results
          tee: false
          time: {horizon: 96, dt_hours: 0.25, start: ...}
    """
    name = str(exp.get("experiment", exp_yaml_path.stem))
    if exp.get("enabled", True) is False:
        logger.info("Skipping experiment: %s (%s)", name, exp_yaml_path.name)
        return

    defaults = exp.get("defaults", {}) if isinstance(exp.get("defaults", {}), dict) else {}

    outputs_dir = _rebase_outputs(Path(defaults.get("outputs_dir", "results")), outputs_root)

    tee_default = bool(defaults.get("tee", False))

    solver_defaults = defaults.get("solver", None)
    if solver_defaults is not None and not isinstance(solver_defaults, dict):
        raise ValueError("defaults.solver must be a dict (e.g. {name: 'highs', options: {...}})")
    solver_name = str((solver_defaults or {}).get("name", "highs"))
    solver_opts = (solver_defaults or {}).get("options", None)

    time_defaults = defaults.get("time", None)
    if time_defaults is not None and not isinstance(time_defaults, dict):
        raise ValueError("defaults.time must be a dict (e.g. {horizon: 96, dt_hours: 0.25, start: '...'})")

    case_yaml_rel = exp.get("case_yaml")
    case_path = resolve_from(exp_yaml_path, case_yaml_rel, root=ROOT)
    if not case_path.exists():
        raise FileNotFoundError(f"Experiment case YAML not found: {case_path}")

    case_name = _case_name_from_yaml(case_path)
    logger.info("Running experiment: %s", name)
    logger.info("  case   : %s (%s)", case_name, case_path)
    logger.info("  outputs: %s", outputs_dir)

    run_case(
        str(case_path),
        outputs_dir=str(outputs_dir),
        tee=tee_default,
        time_override=time_defaults,
        solver=solver_name,
        solver_options=solver_opts,
    )

    logger.info("Single experiment executed.")



def run_all(runset_yaml: str | Path, *, skip_existing: bool = False,
            outputs_root: str | None = None) -> None:
    runset_yaml_path = resolve_from(ROOT, runset_yaml, root=ROOT)
    cfg = load_runset(runset_yaml_path)

    if is_single_experiment(cfg):
        _run_single_experiment(cfg, exp_yaml_path=runset_yaml_path, outputs_root=outputs_root)
        return

    if _is_parent_runset(cfg):
        _run_parent_runset(cfg, parent_yaml_path=runset_yaml_path, skip_existing=skip_existing,
                           outputs_root=outputs_root)
    else:
        _run_single_runset(cfg, runset_yaml_path=runset_yaml_path, skip_existing=skip_existing,
                           outputs_root=outputs_root)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a runset (or parent runset).")
    ap.add_argument("--runset", default=str(ROOT / "cases" / "runset_parent.yaml"), help="Path to runset YAML")
    ap.add_argument("--skip-existing", action="store_true", default=True, help="Skip cases whose output meta.yaml already exists")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Re-run all cases even if output already exists")
    ap.add_argument("--outputs-dir", default=None,
                    help="Redirect ALL outputs to this root (e.g. results_v2), preserving the\n"
                         "per-runset layout. Use to re-run a battery without overwriting results/.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_logging(verbose=args.verbose)
    run_all(args.runset, skip_existing=args.skip_existing, outputs_root=args.outputs_dir)

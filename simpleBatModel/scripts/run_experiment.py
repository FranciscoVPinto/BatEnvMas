from __future__ import annotations

import sys
from pathlib import Path
import argparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_case import run_case
from batEnv.io import load_case_yaml


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

def _case_name_from_yaml(case_yaml_path: Path) -> str:
    try:
        c = load_case_yaml(case_yaml_path)
        name = c.get("case", None)
        return name if isinstance(name, str) and name.strip() else case_yaml_path.stem
    except Exception:
        return case_yaml_path.stem

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge (override wins)."""
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _resolve_from(parent_yaml_path: Path, maybe_rel: str | Path) -> Path:
    """
    Resolve paths relative to the YAML file that references them.
    """
    p = Path(maybe_rel)
    if p.is_absolute():
        return p.resolve()

    cand = (parent_yaml_path.parent / p).resolve()
    if cand.exists():
        return cand

    return (ROOT / p).resolve()

def _is_parent_runset(cfg: dict) -> bool:
    """
    Parent runset format is detected by:
      runset: [runset1.yaml, runset2.yaml, ...]
    i.e., runset is a LIST (not a string).
    """
    return isinstance(cfg.get("runset", None), list)

def _run_single_runset(runset: dict, *, runset_yaml_path: Path):
    runset_name = str(runset.get("runset", runset_yaml_path.stem))

    base_dir = Path(runset.get("cases_base_dir", "cases"))
    if not base_dir.is_absolute():
        base_dir = (ROOT / base_dir).resolve()

    defaults = runset.get("defaults", {}) if isinstance(runset.get("defaults", {}), dict) else {}

    outputs_dir = Path(defaults.get("outputs_dir", "results"))
    if not outputs_dir.is_absolute():
        outputs_dir = (ROOT / outputs_dir).resolve()

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

    case_files = runset.get("cases", [])
    if not isinstance(case_files, list) or not case_files:
        raise ValueError("cases must be a non-empty list")

    print(f"[RUNSET] {runset_name}")
    print(f"[RUNSET] Base dir: {base_dir.resolve()}")
    print(f"[RUNSET] Outputs : {outputs_dir.resolve()}")
    if isinstance(time_defaults, dict) and "horizon" in time_defaults:
        print(f"[RUNSET] Horizon : {time_defaults.get('horizon')}")
    print("")

    for rel in case_files:
        case_path = Path(rel)
        if not case_path.is_absolute():
            case_path = (base_dir / case_path).resolve()

        if not case_path.exists():
            raise FileNotFoundError(f"Case YAML not found: {case_path}")

        case_name = _case_name_from_yaml(case_path)
        if enabled_map.get(case_name, True) is False:
            print(f"[SKIP] {case_name} ({case_path.name})")
            continue

        print(f"[RUN ] {case_name} ({case_path.name})")
        run_case(
            str(case_path),
            outputs_dir=str(outputs_dir),
            tee=tee_default,
            time_override=time_defaults,
            solver=solver_name,
            solver_options=solver_opts,
        )
        print("")

    print("[DONE] All enabled cases executed.")

def _run_parent_runset(parent: dict, *, parent_yaml_path: Path):
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
        runset_yaml_path = _resolve_from(parent_yaml_path, rel)
        if not runset_yaml_path.exists():
            raise FileNotFoundError(f"Child runset YAML not found: {runset_yaml_path}")

        rs = load_runset(runset_yaml_path)

        rs_name = str(rs.get("runset", runset_yaml_path.stem))

        if enabled_map.get(rs_name, True) is False:
            print(f"[SKIP RUNSET] {rs_name} ({runset_yaml_path.name})")
            continue

        # Merge defaults: parent defaults < child defaults
        child_defaults = rs.get("defaults", {}) if isinstance(rs.get("defaults", {}), dict) else {}
        rs["defaults"] = _deep_merge(parent_defaults, child_defaults)

        _run_single_runset(rs, runset_yaml_path=runset_yaml_path)
        print("")

def _is_single_experiment(cfg: dict) -> bool:
    return ("experiment" in cfg) and isinstance(cfg.get("case_yaml", None), (str, Path))

def _run_single_experiment(exp: dict, *, exp_yaml_path: Path):
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
        print(f"[SKIP EXP] {name} ({exp_yaml_path.name})")
        return

    defaults = exp.get("defaults", {}) if isinstance(exp.get("defaults", {}), dict) else {}

    outputs_dir = Path(defaults.get("outputs_dir", "results"))
    if not outputs_dir.is_absolute():
        outputs_dir = (ROOT / outputs_dir).resolve()

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
    case_path = _resolve_from(exp_yaml_path, case_yaml_rel)
    if not case_path.exists():
        raise FileNotFoundError(f"Experiment case YAML not found: {case_path}")

    case_name = _case_name_from_yaml(case_path)
    print(f"[EXPERIMENT] {name}")
    print(f"[EXPERIMENT] case: {case_name} ({case_path})")
    print(f"[EXPERIMENT] outputs: {outputs_dir}")
    print("")

    run_case(
        str(case_path),
        outputs_dir=str(outputs_dir),
        tee=tee_default,
        time_override=time_defaults,
        solver=solver_name,
        solver_options=solver_opts,
    )

    print("")
    print("[DONE] Single experiment executed.")

def run_all(runset_yaml: str | Path):
    runset_yaml_path = _resolve_from(ROOT, runset_yaml)
    cfg = load_runset(runset_yaml_path)

    # If this YAML defines a single experiment, run it and exit.What 
    if _is_single_experiment(cfg):
        _run_single_experiment(cfg, exp_yaml_path=runset_yaml_path)
        return

    if _is_parent_runset(cfg):
        _run_parent_runset(cfg, parent_yaml_path=runset_yaml_path)
    else:
        _run_single_runset(cfg, runset_yaml_path=runset_yaml_path)

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a runset (or parent runset).")
    ap.add_argument("--runset", default=str(ROOT / "cases" / "runset_parent.yaml"), help="Path to runset YAML")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_all(args.runset)
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import yaml

# ---- Spyder-proof: add src/ (and scripts/) to path
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


def run_all(runset_yaml: str | Path):
    runset = load_runset(runset_yaml)
    runset_name = str(runset.get("runset", Path(runset_yaml).stem))

    base_dir = Path(runset.get("cases_base_dir", "cases"))
    if not base_dir.is_absolute():
        base_dir = (ROOT / base_dir).resolve()

    defaults = runset.get("defaults", {}) if isinstance(runset.get("defaults", {}), dict) else {}

    outputs_dir = Path(defaults.get("outputs_dir", "results"))
    if not outputs_dir.is_absolute():
        outputs_dir = (ROOT / outputs_dir).resolve()

    tee_default = bool(defaults.get("tee", False))

    enabled_map = runset.get("enabled", {}) or {}
    if not isinstance(enabled_map, dict):
        raise ValueError("enabled must be a dict mapping case_name -> true/false")

    case_files = runset.get("cases", [])
    if not isinstance(case_files, list) or not case_files:
        raise ValueError("cases must be a non-empty list")

    print(f"[RUNSET] {runset_name}")
    print(f"[RUNSET] Base dir: {base_dir.resolve()}")
    print(f"[RUNSET] Outputs : {outputs_dir.resolve()}")
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
        run_case(str(case_path), outputs_dir=str(outputs_dir), tee=tee_default)
        print("")

    print("[DONE] All enabled cases executed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runset", default=str(ROOT / "cases" / "runset.yaml"), help="Path to runset YAML")
    args = ap.parse_args()
    run_all(args.runset)


if __name__ == "__main__":
    main()

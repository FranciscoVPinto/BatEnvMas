from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml



def add_src_to_path(root: Path) -> None:
    """Make `from batEnv...` imports work without installing the package."""
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logging once per process.

    Idempotent: calling twice does not duplicate handlers.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(level)



def load_yaml_dict(path: str | Path, *, inject_key: Optional[str] = None) -> dict:
    """
    Read a YAML file and require the result to be a dict.

    When `inject_key` is given, the resolved absolute path is stored under that
    key (used by runset/plotset loaders that track the source file).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML at {p} must parse to a dict (got {type(cfg).__name__}).")
    if inject_key is not None:
        cfg[inject_key] = str(p.resolve())
    return cfg



def deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge: dicts are merged key-wise, scalars from `override` win.

    An explicit `null` in `override` DELETES the key from the merged result
    (rather than setting it to None), which is how a child cancels an inherited
    key (e.g. `data.pv_total: ~`).

    Delega na implementação canónica de `batEnv.io.loaders`, a mesma que resolve
    o `extends:` dos ficheiros de caso — assim os overrides de runset e a
    herança de cenários têm garantidamente a MESMA semântica. (Existiam duas
    cópias idênticas; se divergissem, um override de runset passaria a comportar-se
    de forma diferente de um `extends`, o que seria muito difícil de detectar.)
    """
    from batEnv.io.loaders import _deep_merge

    return _deep_merge(base, override)


def resolve_from(yaml_path: Path, maybe_rel: str | Path, *, root: Optional[Path] = None) -> Path:
    """
    Resolve `maybe_rel` against the directory of `yaml_path` first, then `root`.

    Useful when YAMLs reference sibling files via relative paths and we also
    want to fall back to the project root.
    """
    p = Path(maybe_rel)
    if p.is_absolute():
        return p.resolve()

    cand = (yaml_path.parent / p).resolve()
    if cand.exists():
        return cand

    if root is not None:
        return (root / p).resolve()
    return cand



def is_single_experiment(cfg: dict) -> bool:
    """A single-experiment YAML has top-level 'experiment' and 'case_yaml'."""
    return ("experiment" in cfg) and isinstance(cfg.get("case_yaml", None), (str, Path))


def collect_case_files(cfg: dict, base_dir: Path) -> list[Path]:
    """
    Resolve `cases:` and/or `cases_glob:` from a runset/plotset cfg into an
    ordered, de-duplicated list of absolute case YAML paths.

    Supported shapes:
        cases: [a.yaml, b.yaml, ...]
        cases_glob: "b8_pv*.yaml"
        cases_glob: ["b8_pv*.yaml", "other_*.yaml"]

    `cases_glob` patterns are evaluated relative to `base_dir`. Both fields can
    be combined; explicit `cases:` entries are appended first.
    """
    cases_field = cfg.get("cases", []) or []
    glob_field = cfg.get("cases_glob", None)

    if cases_field and not isinstance(cases_field, list):
        raise ValueError("'cases' must be a list of YAML paths")

    out: list[Path] = []
    seen: set[Path] = set()

    for rel in cases_field:
        p = Path(rel)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if p not in seen:
            out.append(p)
            seen.add(p)

    if glob_field is not None:
        patterns = [glob_field] if isinstance(glob_field, str) else list(glob_field)
        for pat in patterns:
            for p in sorted(base_dir.glob(pat)):
                p = p.resolve()
                if p not in seen:
                    out.append(p)
                    seen.add(p)

    if not out:
        raise ValueError("No cases resolved: provide 'cases:' or 'cases_glob:' (or both).")

    return out



def read_house_csvs(case_out_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every results_house_*.csv in `case_out_dir` keyed by house id."""
    dfs: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(case_out_dir.glob("results_house_*.csv")):
        house_id = csv_path.stem.replace("results_house_", "")
        dfs[house_id] = pd.read_csv(csv_path)
    return dfs

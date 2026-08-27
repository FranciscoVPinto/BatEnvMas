"""
Agrega as métricas de um runset (ou de um parent runset) num único CSV.

    %runfile scripts/summarize.py --wdir
    python scripts/summarize.py --runset cases/runset_parent.yaml

Escreve `<outputs_dir>/_summaries/<runset>_summary.csv`, com uma linha por
(caso, casa) mais uma linha `_COMMUNITY` por caso.

NOTA HISTÓRICA: esta versão corrige dois defeitos que tornavam o script
inutilizável com os runsets reais do projecto:

  1. Exigia `cases:` como lista explícita e rebentava com `cases_glob:`, que é
     o que o `full_horizon_sweep` e o `pwl_sweep` usam.
  2. Procurava os resultados em `outputs_dir/<caso>`, ignorando o
     `outputs_subdir` e o sufixo do sweep — ou seja, o layout real
     `outputs_dir/<subdir>/<caso>__<sufixo>` nunca era encontrado.

A resolução de casos e de nomes passa a espelhar exactamente a do
`run_experiment.py`, que é quem escreve os resultados.
"""
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

from _common import (  # noqa: E402
    add_src_to_path,
    collect_case_files,
    deep_merge,
    load_yaml_dict,
    read_house_csvs,
    resolve_from,
    setup_logging,
)

add_src_to_path(ROOT)

import yaml  # noqa: E402

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


def _is_parent_runset(cfg: dict) -> bool:
    return isinstance(cfg.get("runset", None), list)


def _case_name_from_yaml(case_yaml_path: Path) -> str:
    try:
        c = load_case_yaml(case_yaml_path)
    except (OSError, ValueError):
        return case_yaml_path.stem
    name = c.get("case", None)
    return name if isinstance(name, str) and name.strip() else case_yaml_path.stem


def _dt_hours_for(case_out_dir: Path, case_cfg: dict, effective_time: dict) -> float:
    """dt_hours efectivo, preferindo o que ficou registado no `meta.yaml`.

    O `meta.yaml` é a fonte autoritativa: regista o dt com que o caso foi
    realmente resolvido. Só se não existir é que se reconstrói a precedência
    (sweep > defaults do runset > YAML do caso > 1.0).
    """
    meta_path = case_out_dir / "meta.yaml"
    if meta_path.exists():
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            if meta.get("dt_hours"):
                return float(meta["dt_hours"])
        except (OSError, yaml.YAMLError):
            pass
    if effective_time.get("dt_hours"):
        return float(effective_time["dt_hours"])
    time_cfg = case_cfg.get("time", {}) if isinstance(case_cfg.get("time", {}), dict) else {}
    return float(time_cfg.get("dt_hours", 1.0))


def _rows_for_runset(rs: dict, *, runset_yaml_path: Path) -> tuple[list[dict], Path, str]:
    """Recolhe as linhas de métricas de um runset simples."""
    runset_name = str(rs.get("runset", runset_yaml_path.stem))

    base_dir = Path(rs.get("cases_base_dir", "cases"))
    if not base_dir.is_absolute():
        base_dir = (ROOT / base_dir).resolve()

    defaults = rs.get("defaults", {}) if isinstance(rs.get("defaults", {}), dict) else {}
    outputs_dir = Path(defaults.get("outputs_dir", "results"))
    if not outputs_dir.is_absolute():
        outputs_dir = (ROOT / outputs_dir).resolve()

    time_defaults = defaults.get("time", {}) if isinstance(defaults.get("time"), dict) else {}
    enabled_map = rs.get("enabled", {}) or {}

    # Mesma resolução de casos que o run_experiment: aceita `cases` e `cases_glob`.
    case_files = collect_case_files(rs, base_dir)

    # Mesma expansão de sweep que o run_experiment (entrada identidade se ausente).
    sweep_entries = rs.get("sweep") or [{"suffix": "", "overrides": {}}]

    rows: list[dict] = []
    missing: list[str] = []

    for case_path in case_files:
        base_case_name = _case_name_from_yaml(case_path)
        if enabled_map.get(base_case_name, True) is False:
            continue
        case_cfg = load_case_yaml(case_path)

        for sweep in sweep_entries:
            suffix = str(sweep.get("suffix", "")).strip()
            sweep_subdir = sweep.get("outputs_subdir")
            effective_time = dict(time_defaults)
            effective_time.update(sweep.get("time_override") or {})

            run_name = f"{base_case_name}__{suffix}" if suffix else base_case_name
            run_outputs_dir = outputs_dir / sweep_subdir if sweep_subdir else outputs_dir
            out_dir = run_outputs_dir / run_name

            if not out_dir.is_dir():
                missing.append(str(out_dir.relative_to(ROOT) if ROOT in out_dir.parents else out_dir))
                continue

            house_dfs = read_house_csvs(out_dir)
            if not house_dfs:
                missing.append(f"{run_name} (sem results_house_*.csv)")
                continue

            dt_hours = _dt_hours_for(out_dir, case_cfg, effective_time)

            for hid, df in house_dfs.items():
                m = compute_summary_metrics(df, dt_hours=dt_hours)
                m.update(case=run_name, house=hid, scenario=base_case_name,
                         variant=suffix or "-", runset=runset_name)
                rows.append(m)

            df_comm = aggregate_community_timeseries(house_dfs)
            if not df_comm.empty:
                m = compute_summary_metrics(df_comm, dt_hours=dt_hours)
                m.update(compute_community_extra_metrics(df_comm, dt_hours=dt_hours))
                m.update(case=run_name, house=COMMUNITY_ID, scenario=base_case_name,
                         variant=suffix or "-", runset=runset_name)
                rows.append(m)

    if missing:
        logger.warning("%d corrida(s) sem resultados — ignoradas:", len(missing))
        for m in missing[:10]:
            logger.warning("    %s", m)
        if len(missing) > 10:
            logger.warning("    ... e mais %d", len(missing) - 10)

    return rows, outputs_dir, runset_name


def summarize_runset(runset_yaml: str | Path) -> Path:
    runset_yaml_path = resolve_from(ROOT, runset_yaml, root=ROOT)
    cfg = load_runset(runset_yaml_path)

    all_rows: list[dict] = []

    if _is_parent_runset(cfg):
        # Parent: agrega TODOS os runsets filhos activos num único CSV.
        parent_defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {}
        enabled_map = cfg.get("enabled", {}) if isinstance(cfg.get("enabled"), dict) else {}
        out_root = Path(parent_defaults.get("outputs_dir", "results"))
        if not out_root.is_absolute():
            out_root = (ROOT / out_root).resolve()
        name = runset_yaml_path.stem

        for rel in cfg.get("runset", []):
            child_path = resolve_from(runset_yaml_path, rel, root=ROOT)
            if not child_path.exists():
                logger.warning("Runset filho não encontrado: %s", child_path)
                continue
            child = load_runset(child_path)
            child_name = str(child.get("runset", child_path.stem))
            if enabled_map.get(child_name, True) is False:
                logger.info("Runset filho desactivado, ignorado: %s", child_name)
                continue
            child["defaults"] = deep_merge(
                parent_defaults,
                child.get("defaults", {}) if isinstance(child.get("defaults"), dict) else {},
            )
            rows, _, _ = _rows_for_runset(child, runset_yaml_path=child_path)
            logger.info("  %-22s %4d linhas", child_name, len(rows))
            all_rows.extend(rows)
    else:
        all_rows, out_root, name = _rows_for_runset(cfg, runset_yaml_path=runset_yaml_path)

    if not all_rows:
        raise RuntimeError(
            "Nenhum resultado encontrado para sumarizar. Verifique se o runset já "
            "foi corrido e se `defaults.outputs_dir` aponta para a pasta certa."
        )

    summary = pd.DataFrame(all_rows)
    lead = [c for c in ("runset", "scenario", "variant", "case", "house") if c in summary.columns]
    summary = summary[lead + [c for c in summary.columns if c not in lead]]
    summary = summary.set_index(["case", "house"]).sort_index()

    outdir = out_root / "_summaries"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{name}_summary.csv"
    summary.to_csv(outpath)
    logger.info("Escrito: %s  (%d linhas)", outpath, len(summary))
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser(description="Agrega métricas de um runset num CSV.")
    ap.add_argument("--runset", default=str(ROOT / "cases" / "runset_parent.yaml"),
                    help="Caminho do YAML do runset (por omissão, o parent)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Logging DEBUG")
    args = ap.parse_args()
    setup_logging(verbose=args.verbose)
    summarize_runset(args.runset)


if __name__ == "__main__":
    main()

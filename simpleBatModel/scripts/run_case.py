from __future__ import annotations

import sys
from pathlib import Path
import argparse
import datetime as dt
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batEnv.io import (
    load_case_yaml,
    load_series_csv_1col,
    build_tariffs,
    prepare_pv_by_house,
    canonicalize_case_cfg,
    validate_case_cfg_basic,
)
from batEnv.models import SimpleBatteryModel
from batEnv.utils import solve_model, model_to_dataframe


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _abs_from_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _write_1col_csv(path: Path, series: list[float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in series:
            f.write(f"{float(v)}\n")


def run_case(case_yaml: str | Path, outputs_dir: str | Path = "results", tee: bool = False):
    case_yaml_path = _abs_from_root(case_yaml)
    cfg_raw = load_case_yaml(case_yaml_path)

    # Canonicaliza IDs e valida estrutura base cedo
    cfg, canon_warnings = canonicalize_case_cfg(cfg_raw)
    validate_case_cfg_basic(cfg)

    case_name = str(cfg.get("case", None) or Path(case_yaml_path).stem)

    time_cfg = cfg.get("time", {})
    dt_hours = float(time_cfg.get("dt_hours", 1.0))
    T = int(time_cfg.get("horizon", 500))

    data_cfg = cfg.get("data", {})
    loads_cfg = data_cfg.get("loads", {})

    houses_cfg = cfg.get("houses", {})
    house_ids = [str(h) for h in houses_cfg.keys()]

    outputs_dir_path = _abs_from_root(outputs_dir)
    out_case = outputs_dir_path / case_name
    _ensure_dir(out_case)

    c_grid, c_sell = build_tariffs(cfg, T)

    solver_cfg = cfg.get("solver", {})
    solver_name = "highs"
    solver_options = None
    if isinstance(solver_cfg, dict):
        solver_name = solver_cfg.get("name", "highs")
        solver_options = solver_cfg.get("options", None)

    loads_by_house: dict[str, list[float]] = {}
    for hid in house_ids:
        load_path = loads_cfg.get(hid)
        if load_path is None:
            raise ValueError(f"House '{hid}' missing in data.loads mapping")
        loads_by_house[hid] = load_series_csv_1col(_abs_from_root(load_path), T=T)

    pv_by_house, pv_info, pv_debug = prepare_pv_by_house(
        cfg,
        houses=house_ids,
        T=T,
        root=ROOT,
        loads_by_house=loads_by_house,
    )

    preprocess_dir = out_case / "preprocess"
    _ensure_dir(preprocess_dir)
    preprocess_files = {}

    if pv_debug.get("pv_mode") == "shared_alpha":
        pv_total = pv_debug.get("pv_total", None)
        alpha_used = pv_debug.get("alpha_used", None)
        alpha_sum = pv_debug.get("alpha_sum", None)

        if isinstance(pv_total, list):
            p = preprocess_dir / "pv_total.csv"
            _write_1col_csv(p, pv_total)
            preprocess_files["pv_total"] = str(p)

        if isinstance(alpha_sum, list):
            p = preprocess_dir / "alpha_sum.csv"
            _write_1col_csv(p, alpha_sum)
            preprocess_files["alpha_sum"] = str(p)

        if isinstance(alpha_used, dict):
            for hid in house_ids:
                if hid in alpha_used and isinstance(alpha_used[hid], list):
                    p = preprocess_dir / f"alpha_used_{hid}.csv"
                    _write_1col_csv(p, alpha_used[hid])
                    preprocess_files[f"alpha_used_{hid}"] = str(p)

                if hid in pv_by_house and isinstance(pv_by_house[hid], list):
                    p = preprocess_dir / f"pv_alloc_{hid}.csv"
                    _write_1col_csv(p, pv_by_house[hid])
                    preprocess_files[f"pv_alloc_{hid}"] = str(p)

    solved = []
    for hid, hparams in houses_cfg.items():
        hid = str(hid)

        load = loads_by_house[hid]
        pv = pv_by_house[hid]

        batt_cfg = hparams.get("battery", {})
        if not isinstance(batt_cfg, dict):
            raise ValueError(f"houses.{hid}.battery must be a dict")

        # Validações numéricas básicas (para falhar cedo e bem)
        E_init = float(batt_cfg["E_init"])
        E_min = float(batt_cfg["E_min"])
        E_max = float(batt_cfg["E_max"])
        if not (E_min <= E_init <= E_max):
            raise ValueError(f"House {hid}: require E_min <= E_init <= E_max (got {E_min}, {E_init}, {E_max})")

        eta_ch = float(batt_cfg["eta_ch"])
        eta_dis = float(batt_cfg["eta_dis"])
        if not (0 < eta_ch <= 1 and 0 < eta_dis <= 1):
            raise ValueError(f"House {hid}: eta_ch/eta_dis must be in (0,1] (got {eta_ch}, {eta_dis})")

        builder = SimpleBatteryModel(
            dt=dt_hours,
            E_init=E_init,
            E_min=E_min,
            E_max=E_max,
            P_ch_max=float(batt_cfg["P_ch_max"]),
            P_dis_max=float(batt_cfg["P_dis_max"]),
            eta_ch=eta_ch,
            eta_dis=eta_dis,
            P_grid_max=float(batt_cfg["P_grid_max"]),
        )

        m = builder.make_instance(load=load, pv=pv, c_grid=c_grid, c_sell=c_sell)
        results = solve_model(m, solver=solver_name, options=solver_options, tee=tee)
        df = model_to_dataframe(m)

        out_csv = out_case / f"results_house_{hid}.csv"
        df.to_csv(out_csv, index=False)

        solved.append(
            {
                "house": hid,
                "csv": str(out_csv),
                "solver": str(solver_name),
                "termination_condition": str(getattr(results.solver, "termination_condition", "")),
                "status": str(getattr(results.solver, "status", "")),
            }
        )

    meta = {
        "case": case_name,
        "case_yaml": str(case_yaml_path.resolve()),
        "created_at": dt.datetime.now().isoformat(),
        "dt_hours": dt_hours,
        "horizon": T,
        "solver": {"name": solver_name, "options": solver_options},
        "canonicalize_warnings": canon_warnings,
        "pv_preprocess": pv_info,
        "preprocess_files": preprocess_files,
        "houses": solved,
        "notes": "CSVs generated only. Run scripts/make_plots_all.py to generate plots.",
    }

    with (out_case / "meta.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Case '{case_name}' finished. Outputs in: {out_case}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_yaml", help="Path to the YAML case file (e.g., cases/case_1house.yaml)")
    ap.add_argument("--outputs", default="results", help="Base outputs folder (default: results)")
    ap.add_argument("--tee", action="store_true", help="Show solver output")
    args = ap.parse_args()
    run_case(args.case_yaml, outputs_dir=args.outputs, tee=args.tee)


if __name__ == "__main__":
    main()

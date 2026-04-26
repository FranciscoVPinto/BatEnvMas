from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import argparse
import datetime as dt
import yaml
import shutil
import inspect

import pyomo.environ as pyo

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
from batEnv.models import MultiHouseModel
from batEnv.utils.export import multi_model_to_dataframes


# -------------------- small helpers --------------------

def _abs_from_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _pad_or_trunc(arr: list[float], T: int) -> list[float]:
    if len(arr) >= T:
        return arr[:T]
    if not arr:
        return [0.0] * T
    return arr + [arr[-1]] * (T - len(arr))


def _apply_time_override(cfg: dict, time_override: Optional[dict]) -> dict:
    if not time_override:
        return cfg
    out = dict(cfg)
    time_cfg = out.get("time", {}) if isinstance(out.get("time", {}), dict) else {}
    merged = dict(time_cfg)
    for k, v in time_override.items():
        merged[k] = v
    out["time"] = merged
    return out


def _infer_T(cfg: dict) -> int:
    time_cfg = cfg.get("time", {}) if isinstance(cfg.get("time", {}), dict) else {}
    if "horizon" in time_cfg and time_cfg["horizon"] is not None:
        return int(time_cfg["horizon"])

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    loads = data_cfg.get("loads", {}) if isinstance(data_cfg.get("loads", {}), dict) else {}
    for _, rel in loads.items():
        series = load_series_csv_1col(_abs_from_root(rel))
        if series:
            return len(series)
    return 96


def _load_loads(cfg: dict, *, houses: list[str], T: int) -> dict[str, list[float]]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    loads_cfg = data_cfg.get("loads", {})
    if not isinstance(loads_cfg, dict):
        raise ValueError("data.loads must be a dict {house_id: path}")

    loads_by_house: dict[str, list[float]] = {}
    for h in houses:
        if h not in loads_cfg:
            raise KeyError(f"Missing load path for house '{h}' in data.loads")
        fp = _abs_from_root(loads_cfg[h])
        series = load_series_csv_1col(fp)
        loads_by_house[h] = _pad_or_trunc(series, T)
    return loads_by_house


# -------------------- solver handling (quiet + robust) --------------------

def _solver_available_local(name: str) -> bool:
    """
    Avoid noisy Pyomo warnings by pre-checking executables/modules.
    """
    n = (name or "").strip().lower()

    if n == "appsi_highs":
        try:
            import highspy  # noqa: F401
            return True
        except Exception:
            return False

    if n == "highs":
        return shutil.which("highs") is not None

    if n == "glpk":
        return shutil.which("glpsol") is not None

    if n == "cbc":
        return shutil.which("cbc") is not None

    # Fallback: ask Pyomo quietly
    try:
        opt = pyo.SolverFactory(n)
        return bool(opt) and opt.available(exception_flag=False)
    except Exception:
        return False


def _choose_solver_chain(preferred: str) -> list[str]:
    prefs = [preferred] if preferred else []
    for s in ["appsi_highs", "highs", "cbc", "glpk"]:
        if s not in prefs:
            prefs.append(s)
    return prefs


def _solve_model_safe(model: pyo.ConcreteModel, *, solver: str, options: Optional[dict], tee: bool):
    """
    Robust solve wrapper:
      - tries preferred solver then fallbacks that are actually available
      - for APPSI solvers, prevents the "no feasible solution can be loaded" crash
        by setting config.load_solutions=False and only loading if feasible/optimal
    Returns (results, solver_used).
    """
    last_err = None
    chain = [s for s in _choose_solver_chain(solver) if _solver_available_local(s)]
    if not chain:
        raise RuntimeError(
            "No solver available. Install one (recommended on conda):\n"
            "  conda install -c conda-forge highspy\n"
            "and set solver to 'appsi_highs' (or install 'highs' executable)."
        )

    for s in chain:
        try:
            opt = pyo.SolverFactory(s)
            if opt is None:
                continue

            # apply options
            if options and hasattr(opt, "options"):
                for k, v in options.items():
                    opt.options[k] = v

            # HiGHS logging: make tee actually stream output for appsi_highs and ensure HiGHS logs
            if tee and s in ("appsi_highs", "highs"):
                # APPSI HiGHS supports live streaming via config.stream_solver
                if hasattr(opt, "config"):
                    try:
                        if hasattr(opt.config, "stream_solver"):
                            opt.config.stream_solver = True
                    except Exception:
                        pass
                    # If supported, also write a logfile (unless user set one via options)
                    try:
                        if hasattr(opt.config, "logfile") and not getattr(opt.config, "logfile", None):
                            opt.config.logfile = "highs.log"
                    except Exception:
                        pass

                # Standard HiGHS options (work for the direct `highs` interface and are ignored if unsupported)
                if hasattr(opt, "options"):
                    try:
                        opt.options.setdefault("output_flag", True)
                        opt.options.setdefault("log_to_console", True)
                        opt.options.setdefault("log_file", "highs.log")
                        # More frequent MIP progress lines (seconds); harmless for LPs
                        opt.options.setdefault("mip_min_logging_interval", 1)
                    except Exception:
                        pass

            # avoid crash on infeasible for APPSI
            auto_load = True
            if hasattr(opt, "config"):
                try:
                    opt.config.load_solutions = False
                    auto_load = False
                except Exception:
                    pass

            try:
                if tee:
                    import time as _time
                    print(f"[{_time.strftime('%H:%M:%S')}] Starting solve with {s}...")
                results = opt.solve(model, tee=tee, load_solutions=auto_load)
            except TypeError:
                results = opt.solve(model, tee=tee)

            if tee:
                import time as _time
                print(f"[{_time.strftime('%H:%M:%S')}] Solve finished (termination={getattr(results.solver, 'termination_condition', None)})")

            tc = getattr(results.solver, "termination_condition", None)

            if auto_load is False and tc in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
                model.solutions.load_from(results)

            return results, s
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to solve model. Last error: {last_err}")


# -------------------- infeasibility diagnostics (power caps) --------------------

def _top_k(items: list[dict], k: int) -> list[dict]:
    return sorted(items, key=lambda x: x.get("missing_kw", 0.0), reverse=True)[: max(1, k)]


def _diagnose_single_house_power(
    *,
    house: str,
    load: list[float],
    pv: list[float],
    P_grid_max: float,
    P_dis_max: float,
    max_rows: int,
) -> list[dict]:
    """
    Sufficient condition: if Load[t] > PV[t] + P_grid_max + P_dis_max, infeasible at that timestep.
    (If no violations found, infeasibility may still be due to SoC/energy constraints.)
    """
    viol: list[dict] = []
    for t in range(len(load)):
        max_supply = float(pv[t]) + float(P_grid_max) + float(P_dis_max)
        missing = float(load[t]) - max_supply
        if missing > 1e-6:
            viol.append(
                dict(
                    house=house,
                    t=t,
                    load_kw=float(load[t]),
                    pv_kw=float(pv[t]),
                    max_supply_kw=float(max_supply),
                    missing_kw=float(missing),
                    note="Load > PV + P_grid_max + P_dis_max (grid+discharge power caps)",
                )
            )
    return _top_k(viol, max_rows)


def _diagnose_multi_house_power(
    *,
    loads_by_house: dict[str, list[float]],
    pv_by_house: dict[str, list[float]],
    bat_params_by_house: dict[str, dict],
    max_rows: int,
) -> tuple[list[dict], list[dict]]:
    """
    System-level sufficient condition:
      sum_h Load[h,t] > sum_h (PV[h,t] + P_grid_max[h] + P_dis_max[h])  => infeasible

    Per-house sufficient condition:
      Load[h,t] > PV[h,t] + P_grid_max[h] + P_dis_max[h] => infeasible due to local caps
    """
    houses = list(loads_by_house.keys())
    T = len(next(iter(loads_by_house.values())))

    sys_viol: list[dict] = []
    house_viol: list[dict] = []

    for t in range(T):
        total_load = 0.0
        total_pv = 0.0
        total_cap = 0.0

        for h in houses:
            load = float(loads_by_house[h][t])
            pv = float(pv_by_house[h][t])
            P_grid = float((bat_params_by_house.get(h) or {}).get("P_grid_max", 0.0))
            P_dis = float((bat_params_by_house.get(h) or {}).get("P_dis_max", 0.0))

            max_supply_h = pv + P_grid + P_dis
            missing_h = load - max_supply_h
            if missing_h > 1e-6:
                house_viol.append(
                    dict(
                        house=h,
                        t=t,
                        load_kw=load,
                        pv_kw=pv,
                        max_supply_kw=max_supply_h,
                        missing_kw=missing_h,
                        note="House Load > PV + P_grid_max + P_dis_max",
                    )
                )

            total_load += load
            total_pv += pv
            total_cap += pv + P_grid + P_dis

        missing_sys = total_load - total_cap
        if missing_sys > 1e-6:
            sys_viol.append(
                dict(
                    house="__SYSTEM__",
                    t=t,
                    load_kw=total_load,
                    pv_kw=total_pv,
                    max_supply_kw=total_cap,
                    missing_kw=missing_sys,
                    note="System Load > sum(PV + P_grid_max + P_dis_max).",
                )
            )

    return _top_k(sys_viol, max_rows), _top_k(house_viol, max_rows)


def _write_infeas_report(
    *,
    out_dir: Path,
    case_name: str,
    mode: str,
    solver_used: str,
    termination_condition: str,
    debug_cfg: dict,
    lines: list[str],
    model: Optional[pyo.ConcreteModel] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "debug_infeasibility.txt"
    header = [
        f"CASE: {case_name}",
        f"MODE: {mode}",
        f"SOLVER_USED: {solver_used}",
        f"TERMINATION: {termination_condition}",
        "",
        "This case is infeasible (or solver did not return a feasible solution).",
        "Typical causes: grid import cap (P_imp<=P_grid_max), discharge cap (P_dis<=P_dis_max),",
        "SoC bounds (E_min/E_max).",
        "",
    ]
    report_path.write_text("\n".join(header + lines), encoding="utf-8")

    if bool(debug_cfg.get("write_lp", True)) and model is not None:
        try:
            lp_path = out_dir / "debug_model.lp"
            model.write(str(lp_path), io_options={"symbolic_solver_labels": True})
        except Exception:
            pass

    return report_path


# -------------------- compatibility: make_instance signatures --------------------

def _make_multi_instance_compat(builder, **kwargs):
    """
    Call MultiHouseModel.make_instance across versions.

    Common differences:
      - battery_params_by_house vs bat_params_by_house
      - optional 'houses' argument
      - newer versions may accept per-house tariffs dicts, older might expect a shared list
    """
    sig = inspect.signature(builder.make_instance)
    params = sig.parameters

    # rename battery params key if needed
    if "battery_params_by_house" in kwargs and "battery_params_by_house" not in params:
        if "bat_params_by_house" in params:
            kwargs["bat_params_by_house"] = kwargs.pop("battery_params_by_house")

    # inject houses if accepted and missing
    if "houses" in params and "houses" not in kwargs:
        lbh = kwargs.get("loads_by_house")
        if isinstance(lbh, dict):
            kwargs["houses"] = list(lbh.keys())

    # If model doesn't accept dict tariffs, reduce to first house series as a fallback
    try:
        return builder.make_instance(**kwargs)
    except TypeError:
        # second attempt: swap name if opposite exists
        if "bat_params_by_house" in kwargs and "battery_params_by_house" in params:
            alt = dict(kwargs)
            alt["battery_params_by_house"] = alt.pop("bat_params_by_house")
            return builder.make_instance(**alt)
        raise
    except ValueError as e:
        msg = str(e).lower()
        if "tariff" in msg and "length" in msg:
            alt = dict(kwargs)
            for k in ("c_grid", "c_sell"):
                if isinstance(alt.get(k), dict) and alt[k]:
                    first_h = sorted(list(alt[k].keys()))[0]
                    alt[k] = alt[k][first_h]
            return builder.make_instance(**alt)
        raise


# -------------------- main API --------------------

def run_case(
    case_yaml_path: str | Path,
    *,
    outputs_dir: str | Path = "results",
    tee: bool = False,
    time_override: Optional[dict] = None,
    solver: str = "highs",
    solver_options: Optional[dict] = None,
) -> Path:
    case_path = _abs_from_root(case_yaml_path)
    cfg = load_case_yaml(case_path)

    cfg, warnings = canonicalize_case_cfg(cfg)
    cfg = _apply_time_override(cfg, time_override)
    validate_case_cfg_basic(cfg)

    houses = list((cfg.get("houses") or {}).keys())
    houses = [str(h) for h in houses]

    out_root = _abs_from_root(outputs_dir)
    case_name = str(cfg.get("case", case_path.stem))
    case_out_dir = (out_root / case_name).resolve()
    case_out_dir.mkdir(parents=True, exist_ok=True)

    # Debug config (optional)
    debug_cfg = cfg.get("debug", {}) if isinstance(cfg.get("debug", {}), dict) else {}
    debug_cfg.setdefault("write_lp", True)
    debug_cfg.setdefault("max_rows", 30)

    # Horizon and dt
    T = _infer_T(cfg)
    time_cfg = cfg.get("time", {}) if isinstance(cfg.get("time", {}), dict) else {}
    dt_hours = float(time_cfg.get("dt_hours", 1.0))
    start = time_cfg.get("start", None)

    loads_by_house = _load_loads(cfg, houses=houses, T=T)

    pv_by_house, pv_info, _pv_debug = prepare_pv_by_house(
        cfg,
        houses=houses,
        T=T,
        root=ROOT,
        loads_by_house=loads_by_house,
    )

    c_grid, c_sell = build_tariffs(cfg, T, houses=houses, root=ROOT)

    allow_export = bool((cfg.get("grid", {}) or {}).get("allow_export", True))

    # Battery params per house
    bat_params_by_house: dict[str, dict] = {}
    for h in houses:
        hcfg = (cfg.get("houses") or {}).get(h, {}) or {}
        bat_params_by_house[h] = dict(hcfg.get("battery") or {})

    solver_used = None
    termination = None

    # Unified model for 1..N houses (single-house is just N=1)
    mh = MultiHouseModel(
        dt=dt_hours,
        allow_export=allow_export,
    )

    m = _make_multi_instance_compat(
        mh,
        houses=houses,
        loads_by_house=loads_by_house,
        pv_by_house=pv_by_house,
        bat_params_by_house=bat_params_by_house,
        c_grid=c_grid,
        c_sell=c_sell,
    )

    try:
        results, solver_used = _solve_model_safe(m, solver=solver, options=solver_options, tee=tee)
        termination = str(getattr(results.solver, "termination_condition", ""))
        if getattr(results.solver, "termination_condition", None) not in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        ):
            raise RuntimeError(f"Solver returned {termination}")
    except Exception as e:
        sys_viol, house_viol = _diagnose_multi_house_power(
            loads_by_house=loads_by_house,
            pv_by_house=pv_by_house,
            bat_params_by_house=bat_params_by_house,
            max_rows=int(debug_cfg["max_rows"]),
        )

        lines = []
        if sys_viol:
            lines.append("== SYSTEM-LEVEL POWER VIOLATIONS (definitely infeasible) ==")
            for v in sys_viol:
                lines.append(
                    f"- t={v['t']} | Load={v['load_kw']:.3f} | PV={v['pv_kw']:.3f} | "
                    f"MaxSupply≈{v['max_supply_kw']:.3f} | Missing={v['missing_kw']:.3f}"
                )
            lines.append("")
            lines.append("Constraints involved: P_imp<=P_grid_max and P_dis<=P_dis_max.")

        if house_viol:
            if lines:
                lines.append("")
            lines.append("== PER-HOUSE VIOLATIONS ==")
            for v in house_viol:
                lines.append(
                    f"- {v['house']} @ t={v['t']} | Load={v['load_kw']:.3f} | PV={v['pv_kw']:.3f} | "
                    f"MaxSupply≈{v['max_supply_kw']:.3f} | Missing={v['missing_kw']:.3f}"
                )
            lines.append("")
            lines.append("Constraints involved: P_imp<=P_grid_max and P_dis<=P_dis_max.")

        if not sys_viol and not house_viol:
            lines.append("No instantaneous power-cap violation found.")
            lines.append("Infeasibility may be due to inter-temporal SoC/energy constraints (E_min/E_max).")

        report = _write_infeas_report(
            out_dir=case_out_dir,
            case_name=case_name,
            mode="house_indexed_unified",
            solver_used=str(solver_used or solver),
            termination_condition=str(termination or ""),
            debug_cfg=debug_cfg,
            lines=lines,
            model=m,
        )
        raise RuntimeError(f"Infeasible/unsolved case '{case_name}'. See: {report}") from e

    dfs = multi_model_to_dataframes(m)
    for h, df in dfs.items():
        df.to_csv(case_out_dir / f"results_house_{h}.csv", index=False)

    meta = {
        "case": case_name,
        "case_yaml": str(case_path),
        "created_at": dt.datetime.now().isoformat(),
        "dt_hours": dt_hours,
        "horizon": T,
        "start": start,
        "allow_export": allow_export,
        "sharing_enabled": False,
        "model_kind": "house_indexed_unified",
        "warnings": warnings,
        "pv_info": pv_info,
        "solver_used": solver_used,
        "termination_condition": termination,
        "debug": debug_cfg,
    }
    (case_out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    return case_out_dir


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a single case YAML.")
    ap.add_argument("--case", required=True, help="Path to case YAML")
    ap.add_argument("--outputs", default="results", help="Root outputs directory")
    ap.add_argument("--tee", action="store_true", help="Solver tee output")
    ap.add_argument("--solver", default="highs", help="Solver name (appsi_highs/highs/glpk/cbc...)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    run_case(args.case, outputs_dir=args.outputs, tee=bool(args.tee), solver=str(args.solver))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
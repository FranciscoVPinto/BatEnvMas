from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import add_src_to_path, setup_logging  # noqa: E402

add_src_to_path(ROOT)

from batEnv.io import (  # noqa: E402
    build_tariffs,
    canonicalize_case_cfg,
    load_case_yaml,
    load_pv_total,
    load_series_csv_1col,
    prepare_pv_by_house,
    validate_case_cfg_basic,
    validate_case_cfg_schema,
)
from batEnv.models import (  # noqa: E402
    MultiHouseModel,
    MultiHouseModelDegradation,
    MultiHouseModelDegradationPWL,
)
from batEnv.utils.export import multi_model_to_dataframes, extract_pwl_metrics_dataframe  # noqa: E402


logger = logging.getLogger(__name__)


def _abs_from_root(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _pad_or_trunc(arr, T):
    if len(arr) >= T:
        return arr[:T]
    if not arr:
        return [0.0] * T
    return arr + [arr[-1]] * (T - len(arr))


def _apply_time_override(cfg, time_override):
    if not time_override:
        return cfg
    out = dict(cfg)
    time_cfg = out.get("time", {}) if isinstance(out.get("time", {}), dict) else {}
    merged = dict(time_cfg)
    merged.update(time_override)
    out["time"] = merged
    return out


def _infer_T(cfg):
    time_cfg = cfg.get("time", {}) if isinstance(cfg.get("time", {}), dict) else {}
    if "horizon" in time_cfg and time_cfg["horizon"] is not None:
        return int(time_cfg["horizon"])
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    loads = data_cfg.get("loads", {}) if isinstance(data_cfg.get("loads", {}), dict) else {}
    for rel in loads.values():
        series = load_series_csv_1col(_abs_from_root(rel))
        if series:
            return len(series)
    return 96


def _load_loads(cfg, *, houses, T):
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    loads_cfg = data_cfg.get("loads", {})
    if not isinstance(loads_cfg, dict):
        raise ValueError("data.loads must be a dict {house_id: path}")
    loads_by_house = {}
    for h in houses:
        if h not in loads_cfg:
            raise KeyError(f"Missing load path for house '{h}' in data.loads")
        series = load_series_csv_1col(_abs_from_root(loads_cfg[h]))
        loads_by_house[h] = _pad_or_trunc(series, T)
    return loads_by_house


def _solver_available_local(name):
    n = (name or "").strip().lower()
    if n == "appsi_highs":
        try:
            import highspy  # noqa: F401
            return True
        except ImportError:
            return False
    if n == "highs":
        return shutil.which("highs") is not None
    if n == "glpk":
        return shutil.which("glpsol") is not None
    if n == "cbc":
        return shutil.which("cbc") is not None
    try:
        opt = pyo.SolverFactory(n)
    except Exception:
        return False
    return bool(opt) and opt.available(exception_flag=False)


def _choose_solver_chain(preferred):
    chain = [preferred] if preferred else []
    for s in ("appsi_highs", "highs", "cbc", "glpk"):
        if s not in chain:
            chain.append(s)
    return chain


def _configure_highs_logging(opt):
    if hasattr(opt, "config"):
        try:
            if hasattr(opt.config, "stream_solver"):
                opt.config.stream_solver = True
            if hasattr(opt.config, "logfile") and not getattr(opt.config, "logfile", None):
                opt.config.logfile = "highs.log"
        except AttributeError:
            pass
    if hasattr(opt, "options"):
        try:
            opt.options.setdefault("output_flag", True)
            opt.options.setdefault("log_to_console", True)
            opt.options.setdefault("log_file", "highs.log")
            opt.options.setdefault("mip_min_logging_interval", 1)
        except (AttributeError, TypeError):
            pass


def _solve_model_safe(model, *, solver, options, tee):
    chain = [s for s in _choose_solver_chain(solver) if _solver_available_local(s)]
    if not chain:
        raise RuntimeError(
            "No solver available. Install one (recommended on conda):\n"
            "  conda install -c conda-forge highspy\n"
            "and set solver to 'appsi_highs'."
        )
    last_err = None
    for s in chain:
        try:
            opt = pyo.SolverFactory(s)
            if opt is None:
                continue
            if options and hasattr(opt, "options"):
                for k, v in options.items():
                    opt.options[k] = v
            if tee and s in ("appsi_highs", "highs"):
                _configure_highs_logging(opt)
            auto_load = True
            if hasattr(opt, "config"):
                try:
                    opt.config.load_solutions = False
                    auto_load = False
                except AttributeError:
                    pass
            if tee:
                logger.info("Starting solve with %s...", s)
            try:
                results = opt.solve(model, tee=tee, load_solutions=auto_load)
            except TypeError:
                results = opt.solve(model, tee=tee)
            tc = getattr(results.solver, "termination_condition", None)
            if tee:
                logger.info("Solve finished (termination=%s)", tc)
            if not auto_load and tc in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
                model.solutions.load_from(results)
            return results, s
        except Exception as e:  # noqa: BLE001
            logger.debug("Solver %s failed: %s", s, e)
            last_err = e
    raise RuntimeError(f"Failed to solve model. Last error: {last_err}")

def _top_k(items, k):
    return sorted(items, key=lambda x: x.get("missing_kw", 0.0), reverse=True)[:max(1, k)]


def _diagnose_multi_house_power(*, loads_by_house, pv_by_house, bat_params_by_house, max_rows):
    houses = list(loads_by_house.keys())
    T = len(next(iter(loads_by_house.values())))
    sys_viol = []
    house_viol = []
    for t in range(T):
        total_load = total_pv = total_cap = 0.0
        for h in houses:
            load = float(loads_by_house[h][t])
            pv = float(pv_by_house[h][t])
            bat = bat_params_by_house.get(h) or {}
            P_grid = float(bat.get("P_grid_max", 0.0))
            P_dis = float(bat.get("P_dis_max", 0.0))
            max_supply_h = pv + P_grid + P_dis
            missing_h = load - max_supply_h
            if missing_h > 1e-6:
                house_viol.append(dict(house=h, t=t, load_kw=load, pv_kw=pv,
                    max_supply_kw=max_supply_h, missing_kw=missing_h,
                    note="House Load > PV + P_grid_max + P_dis_max"))
            total_load += load
            total_pv += pv
            total_cap += pv + P_grid + P_dis
        missing_sys = total_load - total_cap
        if missing_sys > 1e-6:
            sys_viol.append(dict(house="__SYSTEM__", t=t, load_kw=total_load, pv_kw=total_pv,
                max_supply_kw=total_cap, missing_kw=missing_sys,
                note="System Load > sum(PV + P_grid_max + P_dis_max)."))
    return _top_k(sys_viol, max_rows), _top_k(house_viol, max_rows)


def _format_violations_section(title, viols):
    lines = [title]
    for v in viols:
        lines.append(
            f"- t={v['t']} | Load={v['load_kw']:.3f} | PV={v['pv_kw']:.3f} | "
            f"MaxSupply~{v['max_supply_kw']:.3f} | Missing={v['missing_kw']:.3f}"
        )
    lines.append("")
    lines.append("Constraints: P_imp<=P_grid_max and P_dis<=P_dis_max.")
    return lines


def _write_infeas_report(*, out_dir, case_name, solver_used, termination_condition,
                          debug_cfg, lines, model=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "debug_infeasibility.txt"
    header = [
        f"CASE: {case_name}", "MODE: house_indexed_unified",
        f"SOLVER_USED: {solver_used}", f"TERMINATION: {termination_condition}", "",
        "This case is infeasible (or solver did not return a feasible solution).",
        "Typical causes: P_imp<=P_grid_max, P_dis<=P_dis_max, SoC bounds.", "",
    ]
    report_path.write_text("\n".join(header + lines), encoding="utf-8")
    if bool(debug_cfg.get("write_lp", True)) and model is not None:
        try:
            lp_path = out_dir / "debug_model.lp"
            model.write(str(lp_path), io_options={"symbolic_solver_labels": True})
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug("Failed to write debug_model.lp: %s", e)
    return report_path


def _handle_infeasible(*, case_out_dir, case_name, debug_cfg, loads_by_house,
                        pv_by_house, bat_params_by_house, solver_used, solver,
                        termination, model, original_exc):
    sys_viol, house_viol = _diagnose_multi_house_power(
        loads_by_house=loads_by_house, pv_by_house=pv_by_house,
        bat_params_by_house=bat_params_by_house, max_rows=int(debug_cfg["max_rows"]))
    lines = []
    if sys_viol:
        lines.extend(_format_violations_section(
            "== SYSTEM-LEVEL POWER VIOLATIONS ==", sys_viol))
    if house_viol:
        if lines:
            lines.append("")
        lines.extend(_format_violations_section("== PER-HOUSE VIOLATIONS ==", house_viol))
    if not sys_viol and not house_viol:
        lines.append("No instantaneous power-cap violation found.")
        lines.append("Infeasibility may be due to inter-temporal SoC/energy constraints.")
    report = _write_infeas_report(
        out_dir=case_out_dir, case_name=case_name,
        solver_used=str(solver_used or solver),
        termination_condition=str(termination or ""),
        debug_cfg=debug_cfg, lines=lines, model=model)
    err = RuntimeError(f"Infeasible/unsolved case '{case_name}'. See: {report}")
    err.__cause__ = original_exc
    return err


def run_case(case_yaml_path, *, outputs_dir="results", tee=False,
             time_override=None, solver="highs", solver_options=None):
    case_path = _abs_from_root(case_yaml_path)
    cfg = load_case_yaml(case_path)
    cfg, warnings = canonicalize_case_cfg(cfg)
    cfg = _apply_time_override(cfg, time_override)
    validate_case_cfg_basic(cfg)
    validate_case_cfg_schema(cfg)

    houses = [str(h) for h in (cfg.get("houses") or {}).keys()]
    out_root = _abs_from_root(outputs_dir)
    case_name = str(cfg.get("case", case_path.stem))
    case_out_dir = (out_root / case_name).resolve()
    case_out_dir.mkdir(parents=True, exist_ok=True)

    debug_cfg = cfg.get("debug", {}) if isinstance(cfg.get("debug", {}), dict) else {}
    debug_cfg.setdefault("write_lp", True)
    debug_cfg.setdefault("max_rows", 30)

    T = _infer_T(cfg)
    time_cfg = cfg.get("time", {}) if isinstance(cfg.get("time", {}), dict) else {}
    dt_hours = float(time_cfg.get("dt_hours", 1.0))
    start = time_cfg.get("start", None)

    loads_by_house = _load_loads(cfg, houses=houses, T=T)

    sharing_cfg = cfg.get("sharing") or {}
    sharing_mode = str(sharing_cfg.get("mode", "fixed_alpha"))
    alpha_mode = "optimal" if sharing_mode == "optimal" else "fixed"

    pv_by_house = None
    pv_total = None
    pv_info = {}
    if alpha_mode == "optimal":
        data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
        pv_total = load_pv_total(data_cfg, ROOT, T)
        pv_info = {"mode": "optimal_alpha", "pv_total_sum_kWh": float(sum(pv_total) * dt_hours)}
    else:
        pv_by_house, pv_info, _pv_debug = prepare_pv_by_house(
            cfg, houses=houses, T=T, root=ROOT, loads_by_house=loads_by_house)

    c_grid, c_sell = build_tariffs(cfg, T, houses=houses, root=ROOT)
    allow_export = bool((cfg.get("grid", {}) or {}).get("allow_export", True))

    bat_params_by_house = {
        h: dict(((cfg.get("houses") or {}).get(h, {}) or {}).get("battery") or {})
        for h in houses
    }

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    cyclic_soc = bool(model_cfg.get("cyclic_soc", True))
    lambda_deg = float(model_cfg.get("battery_degradation_eur_per_kwh", 0.0) or 0.0)
    pwl_cfg = model_cfg.get("battery_degradation_pwl", None)

    # Model priority: PWL > linear degradation > base
    if isinstance(pwl_cfg, dict):
        from batEnv.models.multi_house_degradation_pwl import (
            _DEFAULT_LAMBDA_BY_BIN, _DEFAULT_SOC_BREAKPOINTS)
        soc_bkpts = [float(v) for v in pwl_cfg.get("soc_breakpoints", _DEFAULT_SOC_BREAKPOINTS)]
        lam_bins = [float(v) for v in pwl_cfg.get("lambda_by_bin", _DEFAULT_LAMBDA_BY_BIN)]
        lam_per_house_raw = pwl_cfg.get("lambda_by_bin_per_house", None)
        lam_per_house = (
            {str(h): [float(v) for v in vals] for h, vals in lam_per_house_raw.items()}
            if isinstance(lam_per_house_raw, dict) else None
        )
        mh = MultiHouseModelDegradationPWL(
            dt=dt_hours, allow_export=allow_export, cyclic_soc=cyclic_soc,
            soc_breakpoints=soc_bkpts, lambda_by_bin=lam_bins,
            lambda_by_bin_per_house=lam_per_house)
        model_kind = "house_indexed_degradation_pwl"
    elif lambda_deg > 0.0:
        mh = MultiHouseModelDegradation(
            dt=dt_hours, allow_export=allow_export,
            cyclic_soc=cyclic_soc, lambda_deg=lambda_deg)
        model_kind = "house_indexed_degradation"
    else:
        mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=cyclic_soc)
        model_kind = "house_indexed_unified"

    m = mh.make_instance(
        houses=houses, loads_by_house=loads_by_house,
        pv_by_house=pv_by_house, pv_total=pv_total,
        alpha_mode=alpha_mode, bat_params_by_house=bat_params_by_house,
        c_grid=c_grid, c_sell=c_sell)

    solver_used = None
    termination = None
    try:
        results, solver_used = _solve_model_safe(
            m, solver=solver, options=solver_options, tee=tee)
        termination = str(getattr(results.solver, "termination_condition", ""))
        tc = getattr(results.solver, "termination_condition", None)
        if tc not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
            raise RuntimeError(f"Solver returned {termination}")
    except Exception as e:  # noqa: BLE001
        diag_pv_by_house = pv_by_house
        if diag_pv_by_house is None and pv_total is not None:
            n_h = max(1, len(houses))
            diag_pv_by_house = {h: [v / n_h for v in pv_total] for h in houses}
        raise _handle_infeasible(
            case_out_dir=case_out_dir, case_name=case_name, debug_cfg=debug_cfg,
            loads_by_house=loads_by_house, pv_by_house=diag_pv_by_house,
            bat_params_by_house=bat_params_by_house, solver_used=solver_used,
            solver=solver, termination=termination, model=m, original_exc=e) from e

    for h, df in multi_model_to_dataframes(m).items():
        df.to_csv(case_out_dir / f"results_house_{h}.csv", index=False)

    # Export PWL scalar metrics (degradation cost + bin occupancy) when applicable.
    if model_kind == "house_indexed_degradation_pwl":
        pwl_df = extract_pwl_metrics_dataframe(m)
        if not pwl_df.empty:
            pwl_df.to_csv(case_out_dir / "results_pwl_metrics.csv", index=False)

    meta = {
        "case": case_name,
        "case_yaml": str(case_path),
        "created_at": dt.datetime.now().isoformat(),
        "dt_hours": dt_hours,
        "horizon": T,
        "start": start,
        "allow_export": allow_export,
        "cyclic_soc": cyclic_soc,
        "alpha_mode": alpha_mode,
        "battery_degradation_eur_per_kwh": lambda_deg,
        "battery_degradation_pwl": (
            {"soc_breakpoints": mh.soc_breakpoints, "lambda_by_bin": mh.lambda_by_bin}
            if model_kind == "house_indexed_degradation_pwl" else None
        ),
        "sharing_enabled": False,
        "model_kind": model_kind,
        "warnings": warnings,
        "pv_info": pv_info,
        "solver_used": solver_used,
        "termination_condition": termination,
        "debug": debug_cfg,
    }
    (case_out_dir / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return case_out_dir


def _parse_args():
    ap = argparse.ArgumentParser(description="Run a single case YAML.")
    ap.add_argument("--case", required=True, help="Path to case YAML")
    ap.add_argument("--outputs", default="results", help="Root outputs directory")
    ap.add_argument("--tee", action="store_true", help="Solver tee output")
    ap.add_argument("--solver", default="highs",
                    help="Solver name (appsi_highs/highs/glpk/cbc...)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    return ap.parse_args()


def main():
    args = _parse_args()
    setup_logging(verbose=args.verbose)
    run_case(args.case, outputs_dir=args.outputs,
             tee=bool(args.tee), solver=str(args.solver))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
import sys
from pathlib import Path

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
from batEnv.io.loaders import _pad_or_trunc  # noqa: E402
from batEnv.io.resampling import NATIVE_DT_HOURS, resample_factor, resample_series  # noqa: E402
from batEnv.solving import (  # noqa: E402
    ACCEPTABLE_TERMINATIONS,
    handle_infeasible,
    solve_model_safe,
)
from batEnv.models import (  # noqa: E402
    MultiHouseModel,
    MultiHouseModelDegradation,
    MultiHouseModelDegradationPWL,
)
from batEnv.utils.export import multi_model_to_dataframes  # noqa: E402


logger = logging.getLogger(__name__)


def _abs_from_root(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


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


def _tariff_for_house(tariff, h):
    """Extract the tariff series for a single house (handles flat or per-house mapping)."""
    if isinstance(tariff, dict):
        return list(tariff[h])
    return list(tariff)


def _solve_pwl_per_house(
    *, houses, loads_by_house, bat_params_by_house, c_grid, c_sell,
    pv_by_house, dt_hours, allow_export, cyclic_soc, pwl_cfg_dict,
    solver, solver_options, tee, case_name, rolling_cfg=None, case_out_dir=None,
):
    """
    Two-stage linear decomposition for PWL degradation — one sub-problem per house.

    The original MILP PWL formulation adds K binary variables per timestep for bin
    assignment (big-M constraints) on top of the 2 binaries already in the base
    model (charge/discharge mutex, import/export mutex).  The big-M LP relaxation
    is very weak, making branch-and-bound essentially intractable for long horizons.

    This two-stage approach eliminates the bin-assignment binaries entirely:

    Stage 1 — Base MILP (2*T binaries per house, tight LP relaxation):
        Solve the standard MultiHouseModel with no degradation cost.
        Extract the SoC trajectory E[h, 0..T].

    Stage 2 — Augmented MILP (same 2*T binaries, linear degradation term):
        Map E[t-1] → active bin k_t → time-varying lambda_t.
        Re-solve the base model with the additional linear penalty
            sum_t  lambda_t * (P_ch[h,t] + P_dis[h,t]) * dt
        added to the objective.  No new binary variables are introduced.

    The LP relaxation of the base model's no-simultaneous-charge/discharge
    constraints is much tighter than big-M (bounds are physical, not artificial),
    so B&B closes to optimality in a fraction of the time.

    Each house sub-problem is fully independent (fixed-alpha PV allocation), so
    they could in principle be solved in parallel. They are NOT: appsi_highs
    captures stdout/stderr globally (Pyomo tee), which is not thread-safe and
    corrupted solves — producing spurious "infeasibilities" and I/O deadlocks in
    Spyder — when they were run concurrently via a ThreadPoolExecutor. Do not
    reintroduce threading without first isolating the solver's stream handling
    (separate processes would be the safe route).

    Returns
    -------
    house_dfs     : dict[house_id -> pd.DataFrame]
    pwl_metrics   : pd.DataFrame  (all houses concatenated)
    terminations  : dict[house_id -> str]
    solver_used   : str  (from last house solve)
    """
    from batEnv.utils.export import _extract_house_dataframe
    from batEnv.models.multi_house_degradation_pwl import (
        _DEFAULT_LAMBDA_BY_BIN, _DEFAULT_SOC_BREAKPOINTS,
    )
    import pandas as pd

    soc_bkpts = [float(v) for v in pwl_cfg_dict.get("soc_breakpoints", _DEFAULT_SOC_BREAKPOINTS)]
    lam_bins = [float(v) for v in pwl_cfg_dict.get("lambda_by_bin", _DEFAULT_LAMBDA_BY_BIN)]
    K = len(lam_bins)
    lam_per_house_raw = pwl_cfg_dict.get("lambda_by_bin_per_house", None)
    lam_per_house = (
        {str(h): [float(v) for v in vals] for h, vals in lam_per_house_raw.items()}
        if isinstance(lam_per_house_raw, dict) else None
    )

    use_rolling = bool(rolling_cfg and rolling_cfg.get("enabled"))
    if use_rolling:
        from batEnv.utils.rolling_horizon import (
            solve_rolling_horizon, resolve_window_step)

    def _solve_one(h):
        """Two-stage PWL solve for a single house. Thread-safe: builds its own
        Pyomo models and HiGHS instances; shares no mutable state with siblings."""

        # Per-house lambda vector (global default or per-house override)
        lam_h = lam_per_house[str(h)] if (lam_per_house and str(h) in lam_per_house) else lam_bins

        # Battery parameters for bin energy boundaries
        bp = bat_params_by_house.get(h) or {}
        e_min_h = float(bp.get("E_min", 0.0))
        e_max_h = float(bp.get("E_max", 0.0))
        e_init_h = float(bp.get("E_init", 0.0))
        span = max(e_max_h - e_min_h, 1e-9)

        # Energy boundaries for each SoC bin (K bins, K+1 breakpoints)
        e_hi = [e_min_h + soc_bkpts[k + 1] * span for k in range(K)]

        T_h = len(loads_by_house[h])

        # ── Stage 1: base MILP (no degradation) ──────────────────────────────
        if use_rolling:
            win, stp = resolve_window_step(rolling_cfg, T_h)
            logger.info(
                "  [PWL 2-stage] house %s — stage 1 (rolling, window=%d, step=%d, %d timesteps)...",
                h, win, stp, T_h)
            base_m, solver_used_h = solve_rolling_horizon(
                houses=[h],
                loads_by_house={h: loads_by_house[h]},
                pv_by_house={h: pv_by_house[h]},
                bat_params_by_house={h: bp},
                c_grid={h: _tariff_for_house(c_grid, h)},
                c_sell={h: _tariff_for_house(c_sell, h)},
                dt_hours=dt_hours, allow_export=allow_export, cyclic_soc=cyclic_soc,
                window=win, step=stp,
                solve_fn=solve_model_safe, solver=solver,
                solver_options=solver_options, tee=tee,
                case_name=f"{case_name}/{h}", debug_dir=case_out_dir,
            )
            tc_1 = "rolling"
        else:
            logger.info("  [PWL 2-stage] house %s — stage 1 (base MILP, %d timesteps)...", h, T_h)
            base_mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=cyclic_soc)
            base_m = base_mh.make_instance(
                houses=[h],
                loads_by_house={h: loads_by_house[h]},
                bat_params_by_house={h: bp},
                c_grid={h: _tariff_for_house(c_grid, h)},
                c_sell={h: _tariff_for_house(c_sell, h)},
                pv_by_house={h: pv_by_house[h]},
                alpha_mode="fixed",
            )
            results_1, solver_used_h = solve_model_safe(
                base_m, solver=solver, options=solver_options, tee=tee)
            tc_1 = getattr(results_1.solver, "termination_condition", None)
            if tc_1 not in ACCEPTABLE_TERMINATIONS:
                raise RuntimeError(f"Case '{case_name}', house '{h}': stage-1 returned {tc_1}")
            if tc_1 == pyo.TerminationCondition.maxTimeLimit:
                _test = next(iter(base_m.P_imp.values()), None)
                if _test is None or _test.value is None:
                    raise RuntimeError(
                        f"Case '{case_name}', house '{h}': stage-1 hit time limit with no feasible solution.")
                logger.warning("  [PWL 2-stage] house %s stage-1 hit time limit — using best solution.", h)
            logger.info("  [PWL 2-stage] house %s stage-1 done (tc=%s, solver=%s).", h, tc_1, solver_used_h)

        # Extract SoC trajectory; clip to physical bounds for robustness
        E_traj = [e_init_h] + [
            max(e_min_h, min(e_max_h, pyo.value(base_m.E[h, t]) or e_init_h))
            for t in range(1, T_h + 1)
        ]

        # ── Bin assignment from Stage-1 SoC ──────────────────────────────────
        # At start of timestep t, SoC is E_traj[t-1].
        # Bin k is active when E_traj[t-1] < e_hi[k] (last bin catches the rest).
        lambda_t: dict = {}
        z_fixed: dict = {}
        for t in range(1, T_h + 1):
            e_start = E_traj[t - 1]
            k_t = K - 1  # default: last bin
            for k in range(K - 1):
                if e_start < e_hi[k]:
                    k_t = k
                    break
            lambda_t[t] = lam_h[k_t]
            for k in range(K):
                z_fixed[(t, k)] = 1.0 if k == k_t else 0.0

        # ── Stage 2: augmented MILP (same binaries + linear degradation term) ─
        # lambda_t[t] is a Python float (not a Pyomo Param), so the penalty is
        # purely linear — no new binary or continuous variables added.
        if use_rolling:
            win, stp = resolve_window_step(rolling_cfg, T_h)
            logger.info(
                "  [PWL 2-stage] house %s — stage 2 (rolling, window=%d, step=%d)...", h, win, stp)
            deg_m, solver_used_h = solve_rolling_horizon(
                houses=[h],
                loads_by_house={h: loads_by_house[h]},
                pv_by_house={h: pv_by_house[h]},
                bat_params_by_house={h: bp},
                c_grid={h: _tariff_for_house(c_grid, h)},
                c_sell={h: _tariff_for_house(c_sell, h)},
                dt_hours=dt_hours, allow_export=allow_export, cyclic_soc=cyclic_soc,
                window=win, step=stp,
                solve_fn=solve_model_safe, solver=solver,
                solver_options=solver_options, tee=tee,
                lambda_t_by_house={h: lambda_t}, case_name=f"{case_name}/{h}",
                debug_dir=case_out_dir,
            )
            tc_2 = "rolling"
        else:
            logger.info("  [PWL 2-stage] house %s — stage 2 (degradation MILP)...", h)
            deg_mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=cyclic_soc)
            deg_m = deg_mh.make_instance(
                houses=[h],
                loads_by_house={h: loads_by_house[h]},
                bat_params_by_house={h: bp},
                c_grid={h: _tariff_for_house(c_grid, h)},
                c_sell={h: _tariff_for_house(c_sell, h)},
                pv_by_house={h: pv_by_house[h]},
                alpha_mode="fixed",
            )
            base_expr = deg_m.obj.expr
            deg_m.del_component("obj")
            deg_m.obj = pyo.Objective(
                expr=base_expr + pyo.quicksum(
                    lambda_t[t] * (deg_m.P_ch[h, t] + deg_m.P_dis[h, t]) * dt_hours
                    for t in range(1, T_h + 1)
                ),
                sense=pyo.minimize,
            )
            results_2, solver_used_h = solve_model_safe(
                deg_m, solver=solver, options=solver_options, tee=tee)
            tc_2 = getattr(results_2.solver, "termination_condition", None)
            if tc_2 not in ACCEPTABLE_TERMINATIONS:
                raise RuntimeError(f"Case '{case_name}', house '{h}': stage-2 returned {tc_2}")
            if tc_2 == pyo.TerminationCondition.maxTimeLimit:
                _test = next(iter(deg_m.P_imp.values()), None)
                if _test is None or _test.value is None:
                    raise RuntimeError(
                        f"Case '{case_name}', house '{h}': stage-2 hit time limit with no feasible solution.")
                logger.warning("  [PWL 2-stage] house %s stage-2 hit time limit — using best solution.", h)
            logger.info("  [PWL 2-stage] house %s stage-2 done (tc=%s).", h, tc_2)

        df_h = _extract_house_dataframe(deg_m, h)

        # ── PWL metrics (computed from Stage-2 dispatch + Stage-1 bins) ───────
        deg_cost_h = 0.0
        throughput_h = 0.0
        bin_hours_h = {k: 0.0 for k in range(K)}
        for t in range(1, T_h + 1):
            p_ch_t = max(0.0, pyo.value(deg_m.P_ch[h, t]) or 0.0)
            p_dis_t = max(0.0, pyo.value(deg_m.P_dis[h, t]) or 0.0)
            deg_cost_h += lambda_t[t] * (p_ch_t + p_dis_t) * dt_hours
            throughput_h += (p_ch_t + p_dis_t) * dt_hours
            for k in range(K):
                bin_hours_h[k] += z_fixed[(t, k)] * dt_hours

        row: dict = {
            "house": str(h),
            "pwl_degradation_cost_EUR": deg_cost_h,
            "battery_throughput_kWh": throughput_h,
        }
        for k in range(K):
            row[f"bin_hours_{k}"] = bin_hours_h[k]

        return h, df_h, row, f"s1:{tc_1},s2:{tc_2}", solver_used_h

    # ── Execução por casa: SEQUENCIAL, por design ─────────────────────────────
    # As casas são independentes (fixed-alpha) e poderiam correr em paralelo, mas
    # com o appsi_highs isso NÃO é seguro: a captura de stdout/stderr do Pyomo
    # (capture_output / tee) é global e colide entre threads, provocando solves
    # corrompidos (falsas "infeasibilities") e deadlocks de I/O — observado em
    # Spyder/ipykernel. Com rolling horizon cada casa resolve depressa, pelo que
    # o custo de ser sequencial é aceitável. Para paralelizar seria preciso
    # isolar os streams do solver (processos separados, não threads).
    house_dfs: dict = {}
    pwl_rows_map: dict = {}
    terminations: dict = {}
    last_solver = solver

    for h in houses:
        h_res, df, row, term, su = _solve_one(h)
        house_dfs[h_res] = df
        pwl_rows_map[h_res] = row
        terminations[h_res] = term
        last_solver = su

    # Preserve original house order in metrics DataFrame
    pwl_rows = [pwl_rows_map[h] for h in houses]
    pwl_metrics = pd.DataFrame(pwl_rows) if pwl_rows else pd.DataFrame()
    return house_dfs, pwl_metrics, terminations, last_solver


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

    # ── Resampling: load native 15-min data, then down-sample if dt_hours > 0.25 ──
    # All CSV inputs are recorded at NATIVE_DT_HOURS (0.25 h = 15 min).
    # If dt_hours is coarser (e.g. 1.0 h for PWL runs), we must load enough native
    # rows to cover the desired model horizon and then average them down.
    _resample_factor = resample_factor(dt_hours)
    T_native = T * _resample_factor   # rows to load from CSV at 15-min resolution

    loads_by_house = _load_loads(cfg, houses=houses, T=T_native)

    sharing_cfg = cfg.get("sharing") or {}
    sharing_mode = str(sharing_cfg.get("mode", "fixed_alpha"))
    alpha_mode = "optimal" if sharing_mode == "optimal" else "fixed"

    pv_by_house = None
    pv_total = None
    pv_info = {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    if alpha_mode == "optimal":
        pv_total = load_pv_total(data_cfg, ROOT, T_native)
        if _resample_factor > 1:
            pv_total = resample_series(pv_total, _resample_factor)
        pv_info = {"mode": "optimal_alpha", "pv_total_sum_kWh": float(sum(pv_total) * dt_hours)}
    else:
        pv_by_house, pv_info, _pv_debug = prepare_pv_by_house(
            cfg, houses=houses, T=T_native, root=ROOT, loads_by_house=loads_by_house)
        if _resample_factor > 1:
            pv_by_house = {h: resample_series(v, _resample_factor) for h, v in pv_by_house.items()}

    if _resample_factor > 1:
        loads_by_house = {h: resample_series(v, _resample_factor) for h, v in loads_by_house.items()}
        logger.info("Resampled input data from %.2fh to %.2fh (factor=%d, T=%d→%d).",
                    NATIVE_DT_HOURS, dt_hours, _resample_factor, T_native, T)

    c_grid, c_sell = build_tariffs(cfg, T_native, houses=houses, root=ROOT)
    if _resample_factor > 1:
        if isinstance(c_grid, dict):
            c_grid = {h: resample_series(v, _resample_factor) for h, v in c_grid.items()}
            c_sell = {h: resample_series(v, _resample_factor) for h, v in c_sell.items()}
        else:
            c_grid = resample_series(list(c_grid), _resample_factor)
            c_sell = resample_series(list(c_sell), _resample_factor)
    allow_export = bool((cfg.get("grid", {}) or {}).get("allow_export", True))

    bat_params_by_house = {
        h: dict(((cfg.get("houses") or {}).get(h, {}) or {}).get("battery") or {})
        for h in houses
    }

    # ── Validação da Potência Contratada (PC) ────────────────────────────────
    # Em Portugal, a PC deve ser um dos escalões oficiais ERSE.
    # O excedente de PV exportado está limitado à PC (DL 15/2022, art. 23.º),
    # por isso usar um valor fora dos escalões oficiais é um erro de configuração.
    from batEnv.utils.battery_economics import (
        PORTUGUESE_CONTRACTED_POWER_KVA, nearest_contracted_power)
    for h in houses:
        bp = bat_params_by_house.get(h) or {}
        pc = bp.get("P_contracted", bp.get("P_grid_max"))
        if pc is not None:
            pc = float(pc)
            is_official = any(abs(pc - v) < 1e-6 for v in PORTUGUESE_CONTRACTED_POWER_KVA)
            if not is_official:
                suggested = nearest_contracted_power(pc)
                logger.warning(
                    "Casa '%s': P_contracted=%.2f kVA nao e um escalao oficial ERSE. "
                    "Escalao sugerido: %.2f kVA. "
                    "Escaloes validos: %s",
                    h, pc, suggested, PORTUGUESE_CONTRACTED_POWER_KVA,
                )

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    cyclic_soc = bool(model_cfg.get("cyclic_soc", True))
    pwl_cfg = model_cfg.get("battery_degradation_pwl", None)
    rolling_cfg = model_cfg.get("rolling_horizon", None)

    # ── Custo de degradação λ (€/kWh de throughput AC) ──────────────────────
    # Prioridade:
    #   1. model.battery_degradation_eur_per_kwh  — valor explícito no YAML → λ global
    #      (incluindo 0.0 explícito → sem degradação, sem auto-calc)
    #   2. Calculado de parâmetros físicos de cada bateria via Wöhler → λ por casa
    #      (ativado apenas quando a chave está ausente ou é null)
    #   3. Zero — sem penalidade de degradação (modelo base)
    _MISSING = object()
    _raw_lambda = model_cfg.get("battery_degradation_eur_per_kwh", _MISSING)
    _explicit_lambda = _raw_lambda is not _MISSING and _raw_lambda is not None
    lambda_deg = float(_raw_lambda if _explicit_lambda else 0.0)
    lambda_deg_by_house: dict = {}   # vazio → usa lambda_deg global como fallback

    if not _explicit_lambda and lambda_deg == 0.0 and not isinstance(pwl_cfg, dict):
        # Calcular λ individualmente para cada casa a partir dos seus parâmetros físicos
        from batEnv.utils.battery_economics import (
            compute_degradation_cost_per_kwh, degradation_summary)
        for h in houses:
            bp = bat_params_by_house.get(h) or {}
            cost_eur = bp.get("battery_cost_eur")
            N_rated  = bp.get("N_rated_cycles")
            if cost_eur and N_rated:
                try:
                    lam_h = compute_degradation_cost_per_kwh(
                        battery_cost_eur = float(cost_eur),
                        E_max_kwh        = float(bp.get("E_max", 10.0)),
                        E_min_kwh        = float(bp.get("E_min", 0.0)),
                        N_rated_cycles   = float(N_rated),
                        DoD_rated        = float(bp.get("DoD_rated", 0.80)),
                        aging_exponent   = float(bp.get("aging_exponent", 1.50)),
                        eta_ch           = float(bp.get("eta_ch", 0.95)),
                        eta_dis          = float(bp.get("eta_dis", 0.95)),
                    )
                    summary = degradation_summary(
                        battery_cost_eur = float(cost_eur),
                        E_max_kwh        = float(bp.get("E_max", 10.0)),
                        E_min_kwh        = float(bp.get("E_min", 0.0)),
                        N_rated_cycles   = float(N_rated),
                        DoD_rated        = float(bp.get("DoD_rated", 0.80)),
                        aging_exponent   = float(bp.get("aging_exponent", 1.50)),
                        eta_ch           = float(bp.get("eta_ch", 0.95)),
                        eta_dis          = float(bp.get("eta_dis", 0.95)),
                    )
                    lambda_deg_by_house[h] = lam_h
                    logger.info(
                        "Degradação calculada (casa '%s'): λ=%.5f €/kWh | "
                        "N_actual=%.0f ciclos | throughput_vida=%.0f kWh | DoD_real=%.0f%%",
                        h, lam_h, summary["N_actual_cycles"],
                        summary["lifetime_throughput_kwh"],
                        summary["DoD_actual"] * 100,
                    )
                except Exception as deg_err:
                    logger.warning(
                        "Não foi possível calcular λ_deg de parâmetros físicos "
                        "(casa '%s'): %s. A usar λ=0.", h, deg_err)
        # lambda_deg representativo para metadata: média dos valores calculados
        if lambda_deg_by_house:
            lambda_deg = sum(lambda_deg_by_house.values()) / len(lambda_deg_by_house)

    # Model priority: PWL > linear degradation > base
    if isinstance(pwl_cfg, dict):
        model_kind = "house_indexed_degradation_pwl"
        # PWL cases use per-house decomposition — no monolithic model is built here.
        # mh is still created for metadata access (soc_breakpoints, lambda_by_bin).
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
        m = None  # built per-house later
    elif lambda_deg > 0.0 or lambda_deg_by_house:
        mh = MultiHouseModelDegradation(
            dt=dt_hours, allow_export=allow_export,
            cyclic_soc=cyclic_soc, lambda_deg=lambda_deg)
        model_kind = "house_indexed_degradation"
        m = mh.make_instance(
            houses=houses, loads_by_house=loads_by_house,
            pv_by_house=pv_by_house, pv_total=pv_total,
            alpha_mode=alpha_mode, bat_params_by_house=bat_params_by_house,
            c_grid=c_grid, c_sell=c_sell,
            lambda_deg_by_house=lambda_deg_by_house or None)
    else:
        mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=cyclic_soc)
        model_kind = "house_indexed_unified"
        m = mh.make_instance(
            houses=houses, loads_by_house=loads_by_house,
            pv_by_house=pv_by_house, pv_total=pv_total,
            alpha_mode=alpha_mode, bat_params_by_house=bat_params_by_house,
            c_grid=c_grid, c_sell=c_sell)

    # ── PWL decomposed path ──────────────────────────────────────────────────
    # For PWL degradation with fixed-alpha: houses are fully decoupled → solve
    # each house independently (14k binaries each vs 115k monolithic).
    # For PWL with optimal alpha: two-stage — first solve base model to get PV
    # allocation, then fix and decompose per house.
    if model_kind == "house_indexed_degradation_pwl":
        _pwl_pv_by_house = pv_by_house  # may be None for optimal alpha

        if alpha_mode == "optimal":
            # Stage 1: solve base model (no PWL) to get optimal PV allocation.
            # For long horizons where the rest of the pipeline (Stage 2) already
            # uses rolling horizon, apply it here too — the PV-allocation
            # constraint (sum_h PV[h,t] == PV_total[t]) is purely instantaneous,
            # so it composes cleanly with the per-window solve (only the SoC is
            # carried across windows). This avoids the monolithic Stage-1 solve
            # timing out at 6-month/1-year horizons.
            use_rolling_stage1 = bool(rolling_cfg and rolling_cfg.get("enabled"))
            try:
                if use_rolling_stage1:
                    from batEnv.utils.rolling_horizon import (
                        solve_rolling_horizon_pv_allocation, resolve_window_step)
                    win1, stp1 = resolve_window_step(rolling_cfg, T)
                    logger.info(
                        "PWL optimal-alpha: stage 1 — rolling-horizon PV allocation "
                        "(window=%d, step=%d, T=%d)...", win1, stp1, T)
                    _pwl_pv_by_house, _ = solve_rolling_horizon_pv_allocation(
                        houses=houses, loads_by_house=loads_by_house,
                        bat_params_by_house=bat_params_by_house,
                        c_grid=c_grid, c_sell=c_sell, pv_total=pv_total,
                        dt_hours=dt_hours, allow_export=allow_export,
                        cyclic_soc=cyclic_soc, window=win1, step=stp1,
                        solve_fn=solve_model_safe, solver=solver,
                        solver_options=solver_options, tee=tee,
                        case_name=case_name, debug_dir=case_out_dir,
                    )
                else:
                    logger.info("PWL optimal-alpha: stage 1 — solving base model for PV allocation...")
                    base_mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=cyclic_soc)
                    base_m = base_mh.make_instance(
                        houses=houses, loads_by_house=loads_by_house,
                        bat_params_by_house=bat_params_by_house,
                        c_grid=c_grid, c_sell=c_sell,
                        pv_total=pv_total, alpha_mode="optimal",
                    )
                    base_results, base_solver = solve_model_safe(
                        base_m, solver=solver, options=solver_options, tee=False)
                    base_tc = getattr(base_results.solver, "termination_condition", None)
                    if base_tc not in ACCEPTABLE_TERMINATIONS:
                        raise RuntimeError(f"Base model returned {base_tc}")
                    # Extract fixed PV allocation from optimal solution.
                    # max(0.0, ...) guards against tiny negative floats from solver precision.
                    _pwl_pv_by_house = {
                        h: [max(0.0, pyo.value(base_m.PV[h, t]) or 0.0) for t in range(1, T + 1)]
                        for h in houses
                    }
                logger.info("PWL optimal-alpha: stage 1 done. Stage 2 — per-house PWL solve...")
            except Exception as base_e:
                msg = (
                    f"PWL stage-1 PV-allocation solve failed ({base_e!r}) — "
                    f"fell back to EQUAL PV split for case '{case_name}'. "
                    f"Results for this case are NOT a genuine optimal-alpha solution "
                    f"and should be excluded/re-run."
                )
                logger.warning(msg)
                warnings.append(msg)
                _pwl_pv_by_house = {
                    h: [float(pv_total[t]) / max(1, len(houses)) for t in range(T)]
                    for h in houses
                }

        solver_used = None
        try:
            house_dfs, pwl_metrics, terminations, solver_used = _solve_pwl_per_house(
                houses=houses,
                loads_by_house=loads_by_house,
                bat_params_by_house=bat_params_by_house,
                c_grid=c_grid,
                c_sell=c_sell,
                pv_by_house=_pwl_pv_by_house,
                dt_hours=dt_hours,
                allow_export=allow_export,
                cyclic_soc=cyclic_soc,
                pwl_cfg_dict=pwl_cfg,
                solver=solver,
                solver_options=solver_options,
                tee=tee,
                case_name=case_name,
                rolling_cfg=rolling_cfg,
                case_out_dir=case_out_dir,
            )
        except Exception as e:  # noqa: BLE001
            diag_pv = _pwl_pv_by_house or {h: [0.0] * T for h in houses}
            raise handle_infeasible(
                case_out_dir=case_out_dir, case_name=case_name, debug_cfg=debug_cfg,
                loads_by_house=loads_by_house, pv_by_house=diag_pv,
                bat_params_by_house=bat_params_by_house, solver_used=solver_used,
                solver=solver, termination=None, model=None, model_kind=model_kind,
                original_exc=e) from e

        for h, df in house_dfs.items():
            df.to_csv(case_out_dir / f"results_house_{h}.csv", index=False)
        if not pwl_metrics.empty:
            pwl_metrics.to_csv(case_out_dir / "results_pwl_metrics.csv", index=False)

        # Summarise terminations across houses.
        termination = "; ".join(f"{h}:{tc}" for h, tc in terminations.items())

    # ── Standard (monolithic) path ───────────────────────────────────────────
    else:
        solver_used = None
        termination = None
        try:
            results, solver_used = solve_model_safe(
                m, solver=solver, options=solver_options, tee=tee)
            termination = str(getattr(results.solver, "termination_condition", ""))
            tc = getattr(results.solver, "termination_condition", None)
            if tc not in ACCEPTABLE_TERMINATIONS:
                raise RuntimeError(f"Solver returned {termination}")
            if tc == pyo.TerminationCondition.maxTimeLimit:
                _test_var = next(iter(m.P_imp.values()), None)
                _has_solution = _test_var is not None and _test_var.value is not None
                if not _has_solution:
                    raise RuntimeError(
                        f"Case '{case_name}' hit time limit with no feasible solution found.")
                logger.warning("Case '%s' hit time limit — saving best solution found.", case_name)
        except Exception as e:  # noqa: BLE001
            diag_pv_by_house = pv_by_house
            if diag_pv_by_house is None and pv_total is not None:
                n_h = max(1, len(houses))
                diag_pv_by_house = {h: [v / n_h for v in pv_total] for h in houses}
            raise handle_infeasible(
                case_out_dir=case_out_dir, case_name=case_name, debug_cfg=debug_cfg,
                loads_by_house=loads_by_house, pv_by_house=diag_pv_by_house,
                bat_params_by_house=bat_params_by_house, solver_used=solver_used,
                solver=solver, termination=termination, model=m, model_kind=model_kind,
                original_exc=e) from e

        for h, df in multi_model_to_dataframes(m).items():
            df.to_csv(case_out_dir / f"results_house_{h}.csv", index=False)

    # ── Meta ─────────────────────────────────────────────────────────────────
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
        "battery_degradation_eur_per_kwh_by_house": lambda_deg_by_house or None,
        "battery_degradation_pwl": (
            {"soc_breakpoints": mh.soc_breakpoints, "lambda_by_bin": mh.lambda_by_bin}
            if model_kind == "house_indexed_degradation_pwl" else None
        ),
        "rolling_horizon": (
            dict(rolling_cfg) if (rolling_cfg and rolling_cfg.get("enabled")) else None
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

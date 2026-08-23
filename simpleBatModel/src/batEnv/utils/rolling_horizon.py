"""
Solve de horizonte deslizante (receding / rolling horizon) para o modelo
house-indexed (`MultiHouseModel`), em modo fixed-alpha.

Motivacao
---------
O modelo base tem um binario `y[h,t]` por timestep e por casa. Em horizontes
longos (ex.: 1 ano a 15 min => 35040 timesteps) o branch-and-bound torna-se
caro. O horizonte deslizante divide o horizonte em janelas sobrepostas:

  - cada janela tem `window` timesteps (commit + look-ahead);
  - resolve-se a janela, commitam-se os primeiros `step` timesteps;
  - o SoC no fim do trecho commitado e transportado para a janela seguinte;
  - a janela avanca `step` timesteps.

O trecho final `window - step` (look-ahead) evita a miopia de fim-de-janela
(ex.: esvaziar a bateria mesmo antes da fronteira), porque a decisao no
instante de commit ja "ve" o futuro proximo.

Natureza
--------
E uma APROXIMACAO, nao o otimo global: as decisoes sao miopes para alem do
look-ahead. Para baterias domesticas pequenas com ciclo diario o desvio e
pequeno, pois o acoplamento intertemporal raramente excede alguns dias. O solve
monolitico exacto continua a ser o default; o rolling horizon e opt-in via
`model.rolling_horizon.enabled`.

A restricao de SoC ciclico global (E[T] >= E_init) e imposta apenas na ultima
janela. As janelas intermedias nao tem restricao terminal de SoC; o look-ahead
mantem o SoC de fronteira sensato.

A funcao devolve um `MultiHouseModel` completo (horizonte inteiro) com os
valores das variaveis ja populados a partir das janelas commitadas. Esse modelo
NAO e resolvido como monolito — serve apenas de contentor, para que a extracao a
jusante (`_extract_house_dataframe`) funcione sem alteracoes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import pyomo.environ as pyo

logger = logging.getLogger("batEnv.rolling_horizon")

_ACCEPTABLE = (
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.feasible,
    pyo.TerminationCondition.maxTimeLimit,
)

_VAR_NAMES = ("P_ch", "P_dis", "P_imp", "P_exp", "P_curt")


def _slice(series: Sequence[float], a: int, b: int) -> list:
    """Corte inclusivo [a, b] (indices 1-based) de uma lista 0-based."""
    return list(series)[a - 1:b]


def _v(x, default: float = 0.0) -> float:
    val = pyo.value(x, exception=False)
    return float(val) if val is not None else float(default)


def resolve_window_step(
    rolling_cfg: Optional[Mapping[str, Any]],
    T: int,
    *,
    default_window: int = 1344,
    default_step: int = 672,
) -> Tuple[int, int]:
    """Determina (window, step) a partir do cfg, com defaults e validacao.

    Defaults: window=1344, step=672 (a dt=0.25 h => ~14 dias de janela com ~7
    dias de commit). Se window >= T, reduz-se a um unico solve (exacto).
    """
    cfg = rolling_cfg or {}
    window = int(cfg.get("window", default_window))
    step = int(cfg.get("step", default_step))
    if window < 2:
        raise ValueError(f"rolling_horizon.window deve ser >= 2, obtido {window}")
    if not (1 <= step <= window):
        raise ValueError(
            f"rolling_horizon.step deve verificar 1 <= step <= window, "
            f"obtido step={step}, window={window}"
        )
    # Nunca faz sentido janela maior que o horizonte.
    window = min(window, T)
    step = min(step, window)
    return window, step


def solve_rolling_horizon(
    *,
    houses: Sequence[str],
    loads_by_house: Mapping[str, Sequence[float]],
    pv_by_house: Mapping[str, Sequence[float]],
    bat_params_by_house: Mapping[str, Mapping[str, Any]],
    c_grid: Mapping[str, Sequence[float]],
    c_sell: Mapping[str, Sequence[float]],
    dt_hours: float,
    allow_export: bool,
    cyclic_soc: bool,
    window: int,
    step: int,
    solve_fn: Callable,
    solver: str,
    solver_options: Optional[dict],
    tee: bool = False,
    lambda_t_by_house: Optional[Mapping[str, Mapping[int, float]]] = None,
    case_name: str = "",
    debug_dir: Optional[Any] = None,
):
    """
    Resolve o modelo fixed-alpha house-indexed por janelas deslizantes.

    Parametros espelham `MultiHouseModel.make_instance` (apenas fixed alpha).
    `lambda_t_by_house[h][t]` (t global em 1..T) adiciona o termo linear de
    degradacao  lambda * (P_ch + P_dis) * dt  ao objetivo (usado no stage 2 PWL).

    Returns
    -------
    full_m      : pyo.ConcreteModel  (horizonte inteiro, valores populados)
    last_solver : str
    """
    from batEnv.models.multi_house import MultiHouseModel

    houses = [str(h) for h in houses]
    T = len(next(iter(loads_by_house.values())))
    if not (1 <= step <= window):
        raise ValueError(f"step/window invalidos: step={step}, window={window}")

    # Trajetorias commitadas, t global em 1..T
    committed: Dict[str, Dict[str, Dict[int, float]]] = {
        h: {vn: {} for vn in _VAR_NAMES} for h in houses
    }
    committed_E: Dict[str, Dict[int, float]] = {h: {} for h in houses}
    committed_y: Dict[str, Dict[int, float]] = {h: {} for h in houses}

    global_E_init = {
        h: float((bat_params_by_house.get(h) or {}).get("E_init", 0.0)) for h in houses
    }
    e_min = {h: float((bat_params_by_house.get(h) or {}).get("E_min", 0.0)) for h in houses}
    e_max = {h: float((bat_params_by_house.get(h) or {}).get("E_max", 0.0)) for h in houses}
    carried_E = dict(global_E_init)

    last_solver = solver
    a = 1
    n_win = 0
    safe_name = str(case_name or "case").replace("/", "__")
    window_log_path = None
    if debug_dir is not None:
        window_log_path = Path(debug_dir) / f"highs_rolling_{safe_name}.log"
        window_log_path.parent.mkdir(parents=True, exist_ok=True)
    def _soc_at(h, t):
        """SoC at global timestep t (t=0 -> original E_init; t>0 must already
        be committed by a previous window)."""
        return global_E_init[h] if t <= 0 else committed_E[h][t]

    while a <= T:
        b = min(a + window - 1, T)
        is_final = (b == T)
        commit_end = b if is_final else min(a + step - 1, T)
        natural_wlen = b - a + 1
        needs_anchor = is_final and natural_wlen < window
        n_win += 1

        # The FINAL window, when it would otherwise be truncated (T not an
        # exact multiple of step), is anchored backward and *grown* until
        # the solve is feasible. A short tail window (e.g. the last ~8 days
        # of a year) gives the cyclic SoC constraint (E[T] >= E_init) very
        # little room to recharge the battery — this can be a genuine
        # physical infeasibility (not enough PV surplus in that stretch of
        # winter for that house), not just an artifact of truncation. So we
        # search for the smallest lookback (in increments of `step`) that
        # makes it feasible, capped at the full horizon [1, T]. Only the
        # still-uncommitted tail [a, commit_end] gets committed from the
        # winning solve; the look-back portion [a_solve, a-1] — already
        # committed by earlier windows — is resolved again purely to give
        # the tail more context/energy to work with.
        lookback_len = window if needs_anchor else natural_wlen
        attempt = 0
        while True:
            attempt += 1
            a_solve = max(1, b - lookback_len + 1) if needs_anchor else a
            wlen = b - a_solve + 1

            # Parametros da bateria com E_init = SoC no inicio da janela resolvida.
            wbp = {}
            for h in houses:
                bp = dict(bat_params_by_house.get(h) or {})
                bp["E_init"] = _soc_at(h, a_solve - 1)
                wbp[h] = bp

            mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=False)
            wm = mh.make_instance(
                houses=houses,
                loads_by_house={h: _slice(loads_by_house[h], a_solve, b) for h in houses},
                bat_params_by_house=wbp,
                c_grid={h: _slice(c_grid[h], a_solve, b) for h in houses},
                c_sell={h: _slice(c_sell[h], a_solve, b) for h in houses},
                pv_by_house={h: _slice(pv_by_house[h], a_solve, b) for h in houses},
                alpha_mode="fixed",
            )

            # SoC ciclico global: so na ultima janela.
            if is_final and cyclic_soc:
                wm.rh_cyclic = pyo.Constraint(
                    wm.H, rule=lambda mm, h, _w=wlen: mm.E[h, _w] >= global_E_init[h]
                )

            # Penalidade linear de degradacao (stage 2 PWL), se fornecida.
            if lambda_t_by_house is not None:
                base_expr = wm.obj.expr
                wm.del_component("obj")
                wm.obj = pyo.Objective(
                    expr=base_expr + pyo.quicksum(
                        float(lambda_t_by_house[h].get(a_solve + i, 0.0))
                        * (wm.P_ch[h, i + 1] + wm.P_dis[h, i + 1]) * dt_hours
                        for h in houses for i in range(wlen)
                    ),
                    sense=pyo.minimize,
                )

            # window_log_path is overwritten every attempt, so on a final
            # failure it holds the raw HiGHS log for exactly that attempt.
            results, su = solve_fn(
                wm, solver=solver, options=solver_options, tee=tee,
                log_file=window_log_path,
            )
            tc = getattr(results.solver, "termination_condition", None)
            if tc in _ACCEPTABLE:
                break  # sucesso — sai do loop de tentativas

            can_grow = needs_anchor and a_solve > 1
            if not can_grow:
                status = getattr(results.solver, "status", None)
                message = (
                    getattr(results.solver, "termination_message", None)
                    or getattr(results.solver, "message", None)
                )
                hint = f" | HiGHS log: {window_log_path}" if window_log_path is not None else ""
                if debug_dir is not None:
                    try:
                        lp_path = Path(debug_dir) / f"debug_rolling_{safe_name}_win{a_solve}-{b}.lp"
                        wm.write(str(lp_path), io_options={"symbolic_solver_labels": True})
                        hint += f" | LP dumped: {lp_path}"
                    except Exception as dump_err:  # noqa: BLE001
                        logger.debug("Failed to dump rolling-window LP: %s", dump_err)
                anchor_note = f" (resolvida como [{a_solve},{b}], tentativa {attempt})" if a_solve != a else ""
                raise RuntimeError(
                    f"Case '{case_name}': janela rolling [{a},{b}]{anchor_note} devolveu {tc} "
                    f"(status={status}, message={message}){hint}")

            # Ainda pode crescer: alarga o lookback e tenta de novo.
            logger.warning(
                "  [rolling] %s: janela final [%d,%d] com lookback de %.1f dias (%d passos) "
                "ainda %s — a alargar para tras (tentativa %d).",
                case_name or "case", a, b, lookback_len / 96.0, lookback_len, tc, attempt + 1,
            )
            lookback_len = min(lookback_len + step, b)

        last_solver = su
        if tc == pyo.TerminationCondition.maxTimeLimit:
            probe = pyo.value(wm.P_imp[houses[0], 1], exception=False)
            if probe is None:
                raise RuntimeError(
                    f"Case '{case_name}': janela rolling [{a},{b}] sem solucao "
                    f"viavel no limite de tempo.")
            logger.warning(
                "  [rolling] %s: janela [%d,%d] atingiu o limite de tempo — a usar a melhor solucao.",
                case_name or "case", a, b)
        elif needs_anchor and a_solve != max(1, b - window + 1):
            logger.info(
                "  [rolling] %s: janela final [%d,%d] precisou de lookback alargado ate [%d,%d] "
                "(%.1f dias) para ser viavel (tentativa %d).",
                case_name or "case", a, b, a_solve, b, wlen / 96.0, attempt,
            )

        # Commit [a, commit_end]. Local index is relative to a_solve (the
        # actual start of the solved window), not a (the commit frontier) —
        # they only coincide when the window wasn't anchored backward.
        for h in houses:
            for gt in range(a, commit_end + 1):
                i = gt - a_solve + 1  # indice local 1..wlen
                committed[h]["P_ch"][gt] = max(0.0, _v(wm.P_ch[h, i]))
                committed[h]["P_dis"][gt] = max(0.0, _v(wm.P_dis[h, i]))
                committed[h]["P_imp"][gt] = max(0.0, _v(wm.P_imp[h, i]))
                committed[h]["P_exp"][gt] = max(0.0, _v(wm.P_exp[h, i]))
                committed[h]["P_curt"][gt] = max(0.0, _v(wm.P_curt[h, i]))
                e_val = min(e_max[h], max(e_min[h], _v(wm.E[h, i], carried_E[h])))
                committed_E[h][gt] = e_val
                committed_y[h][gt] = float(round(_v(wm.y[h, i])))
            carried_E[h] = committed_E[h][commit_end]

        a = commit_end + 1

    logger.info(
        "  [rolling] %s: %d janelas resolvidas (window=%d, step=%d, T=%d).",
        case_name or "case", n_win, window, step, T)

    # Shell de horizonte inteiro (NAO resolvido) com valores populados.
    full_mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=False)
    full_m = full_mh.make_instance(
        houses=houses,
        loads_by_house={h: list(loads_by_house[h]) for h in houses},
        bat_params_by_house={h: dict(bat_params_by_house.get(h) or {}) for h in houses},
        c_grid={h: list(c_grid[h]) for h in houses},
        c_sell={h: list(c_sell[h]) for h in houses},
        pv_by_house={h: list(pv_by_house[h]) for h in houses},
        alpha_mode="fixed",
    )
    for h in houses:
        full_m.E[h, 0].set_value(global_E_init[h])
        for gt in range(1, T + 1):
            full_m.P_ch[h, gt].set_value(committed[h]["P_ch"][gt])
            full_m.P_dis[h, gt].set_value(committed[h]["P_dis"][gt])
            full_m.P_imp[h, gt].set_value(committed[h]["P_imp"][gt])
            full_m.P_exp[h, gt].set_value(committed[h]["P_exp"][gt])
            full_m.P_curt[h, gt].set_value(committed[h]["P_curt"][gt])
            full_m.E[h, gt].set_value(committed_E[h][gt])
            full_m.y[h, gt].set_value(committed_y[h][gt])

    return full_m, last_solver


def solve_rolling_horizon_pv_allocation(
    *,
    houses: Sequence[str],
    loads_by_house: Mapping[str, Sequence[float]],
    bat_params_by_house: Mapping[str, Mapping[str, Any]],
    c_grid: Mapping[str, Sequence[float]],
    c_sell: Mapping[str, Sequence[float]],
    pv_total: Sequence[float],
    dt_hours: float,
    allow_export: bool,
    cyclic_soc: bool,
    window: int,
    step: int,
    solve_fn: Callable,
    solver: str,
    solver_options: Optional[dict],
    tee: bool = False,
    case_name: str = "",
    debug_dir: Optional[Any] = None,
) -> Tuple[Dict[str, list], str]:
    """
    Resolve o modelo base com `alpha_mode='optimal'` por janelas deslizantes,
    devolvendo apenas a alocacao de PV por casa (`PV[h,t]`) para todo o
    horizonte T.

    Motivacao
    ---------
    Usado como Stage 1 do caminho PWL com `alpha_mode='optimal'`
    (`run_case._solve_pwl_per_house`), que ate agora resolvia sempre o modelo
    base *monoliticamente* para obter a alocacao de PV, mesmo quando o resto
    do pipeline (Stage 2, degradacao PWL) ja usava rolling horizon a 6
    meses/1 ano. A horizontes longos esse solve monolitico pode exceder o
    tempo limite ou falhar, o que antes caia num fallback silencioso para
    divisao igual de PV entre casas (ver `run_case.py`). Esta funcao fecha
    essa lacuna aplicando o mesmo horizonte deslizante tambem a Stage 1.

    A restricao de alocacao (`sum_h PV[h,t] == PV_total[t]`) e puramente
    instantanea (nao tem acoplamento inter-temporal entre casas), por isso
    pode ser resolvida janela a janela exatamente como o modelo fixed-alpha
    — o unico estado transportado entre janelas continua a ser o SoC de
    cada bateria.
    """
    from batEnv.models.multi_house import MultiHouseModel

    houses = [str(h) for h in houses]
    T = len(next(iter(loads_by_house.values())))
    if not (1 <= step <= window):
        raise ValueError(f"step/window invalidos: step={step}, window={window}")

    committed_pv: Dict[str, Dict[int, float]] = {h: {} for h in houses}
    committed_E: Dict[str, Dict[int, float]] = {h: {} for h in houses}

    global_E_init = {
        h: float((bat_params_by_house.get(h) or {}).get("E_init", 0.0)) for h in houses
    }
    e_min = {h: float((bat_params_by_house.get(h) or {}).get("E_min", 0.0)) for h in houses}
    e_max = {h: float((bat_params_by_house.get(h) or {}).get("E_max", 0.0)) for h in houses}
    carried_E = dict(global_E_init)

    last_solver = solver
    a = 1
    n_win = 0
    safe_name = str(case_name or "case").replace("/", "__")
    window_log_path = None
    if debug_dir is not None:
        window_log_path = Path(debug_dir) / f"highs_rolling_pvalloc_{safe_name}.log"
        window_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _soc_at(h, t):
        return global_E_init[h] if t <= 0 else committed_E[h][t]

    while a <= T:
        b = min(a + window - 1, T)
        is_final = (b == T)
        commit_end = b if is_final else min(a + step - 1, T)
        natural_wlen = b - a + 1
        needs_anchor = is_final and natural_wlen < window
        n_win += 1

        lookback_len = window if needs_anchor else natural_wlen
        attempt = 0
        while True:
            attempt += 1
            a_solve = max(1, b - lookback_len + 1) if needs_anchor else a
            wlen = b - a_solve + 1

            wbp = {}
            for h in houses:
                bp = dict(bat_params_by_house.get(h) or {})
                bp["E_init"] = _soc_at(h, a_solve - 1)
                wbp[h] = bp

            mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=False)
            wm = mh.make_instance(
                houses=houses,
                loads_by_house={h: _slice(loads_by_house[h], a_solve, b) for h in houses},
                bat_params_by_house=wbp,
                c_grid={h: _slice(c_grid[h], a_solve, b) for h in houses},
                c_sell={h: _slice(c_sell[h], a_solve, b) for h in houses},
                pv_total=_slice(pv_total, a_solve, b),
                alpha_mode="optimal",
            )

            if is_final and cyclic_soc:
                wm.rh_cyclic = pyo.Constraint(
                    wm.H, rule=lambda mm, h, _w=wlen: mm.E[h, _w] >= global_E_init[h]
                )

            results, su = solve_fn(
                wm, solver=solver, options=solver_options, tee=tee,
                log_file=window_log_path,
            )
            tc = getattr(results.solver, "termination_condition", None)
            if tc in _ACCEPTABLE:
                break

            can_grow = needs_anchor and a_solve > 1
            if not can_grow:
                status = getattr(results.solver, "status", None)
                message = (
                    getattr(results.solver, "termination_message", None)
                    or getattr(results.solver, "message", None)
                )
                hint = f" | HiGHS log: {window_log_path}" if window_log_path is not None else ""
                anchor_note = f" (resolvida como [{a_solve},{b}], tentativa {attempt})" if a_solve != a else ""
                raise RuntimeError(
                    f"Case '{case_name}': janela rolling (PV alloc) [{a},{b}]{anchor_note} "
                    f"devolveu {tc} (status={status}, message={message}){hint}")

            logger.warning(
                "  [rolling-pvalloc] %s: janela final [%d,%d] com lookback de %.1f dias "
                "(%d passos) ainda %s — a alargar para tras (tentativa %d).",
                case_name or "case", a, b, lookback_len / 96.0, lookback_len, tc, attempt + 1,
            )
            lookback_len = min(lookback_len + step, b)

        last_solver = su
        if tc == pyo.TerminationCondition.maxTimeLimit:
            probe = pyo.value(wm.PV[houses[0], 1], exception=False)
            if probe is None:
                raise RuntimeError(
                    f"Case '{case_name}': janela rolling (PV alloc) [{a},{b}] sem solucao "
                    f"viavel no limite de tempo.")
            logger.warning(
                "  [rolling-pvalloc] %s: janela [%d,%d] atingiu o limite de tempo — "
                "a usar a melhor solucao.", case_name or "case", a, b)
        elif needs_anchor and a_solve != max(1, b - window + 1):
            logger.info(
                "  [rolling-pvalloc] %s: janela final [%d,%d] precisou de lookback alargado "
                "ate [%d,%d] (%.1f dias) para ser viavel (tentativa %d).",
                case_name or "case", a, b, a_solve, b, wlen / 96.0, attempt,
            )

        for h in houses:
            for gt in range(a, commit_end + 1):
                i = gt - a_solve + 1
                committed_pv[h][gt] = max(0.0, _v(wm.PV[h, i]))
                e_val = min(e_max[h], max(e_min[h], _v(wm.E[h, i], carried_E[h])))
                committed_E[h][gt] = e_val
            carried_E[h] = committed_E[h][commit_end]

        a = commit_end + 1

    logger.info(
        "  [rolling-pvalloc] %s: %d janelas resolvidas (window=%d, step=%d, T=%d).",
        case_name or "case", n_win, window, step, T)

    pv_by_house = {h: [committed_pv[h][t] for t in range(1, T + 1)] for h in houses}
    return pv_by_house, last_solver

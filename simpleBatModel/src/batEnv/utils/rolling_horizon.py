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
    while a <= T:
        b = min(a + window - 1, T)
        is_final = (b == T)
        commit_end = b if is_final else min(a + step - 1, T)
        wlen = b - a + 1
        n_win += 1

        # Parametros da bateria com E_init = SoC transportado.
        wbp = {}
        for h in houses:
            bp = dict(bat_params_by_house.get(h) or {})
            bp["E_init"] = carried_E[h]
            wbp[h] = bp

        mh = MultiHouseModel(dt=dt_hours, allow_export=allow_export, cyclic_soc=False)
        wm = mh.make_instance(
            houses=houses,
            loads_by_house={h: _slice(loads_by_house[h], a, b) for h in houses},
            bat_params_by_house=wbp,
            c_grid={h: _slice(c_grid[h], a, b) for h in houses},
            c_sell={h: _slice(c_sell[h], a, b) for h in houses},
            pv_by_house={h: _slice(pv_by_house[h], a, b) for h in houses},
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
                    float(lambda_t_by_house[h].get(a + i, 0.0))
                    * (wm.P_ch[h, i + 1] + wm.P_dis[h, i + 1]) * dt_hours
                    for h in houses for i in range(wlen)
                ),
                sense=pyo.minimize,
            )

        results, su = solve_fn(wm, solver=solver, options=solver_options, tee=tee)
        last_solver = su
        tc = getattr(results.solver, "termination_condition", None)
        if tc not in _ACCEPTABLE:
            raise RuntimeError(
                f"Case '{case_name}': janela rolling [{a},{b}] devolveu {tc}")
        if tc == pyo.TerminationCondition.maxTimeLimit:
            probe = pyo.value(wm.P_imp[houses[0], 1], exception=False)
            if probe is None:
                raise RuntimeError(
                    f"Case '{case_name}': janela rolling [{a},{b}] sem solucao "
                    f"viavel no limite de tempo.")
            logger.warning(
                "  [rolling] %s: janela [%d,%d] atingiu o limite de tempo — a usar a melhor solucao.",
                case_name or "case", a, b)

        # Commit [a, commit_end].
        for h in houses:
            for gt in range(a, commit_end + 1):
                i = gt - a + 1  # indice local 1..wlen
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

"""
Diagnóstico de infeasibilidade.

Antes de aceitar uma falha do solver, verifica se o cenário é infeasível *por
construção* — isto é, se a carga excede, nalgum instante, a soma do PV
disponível, da potência contratada e da potência de descarga. Isso separa erros
de dados/parâmetros de problemas do lado do solver, e é o que a Secção 3.5 da
dissertação descreve como "infeasibility diagnostics".

Movido de `scripts/run_case.py` sem alterações de comportamento.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("batEnv.solving")


def _top_k(items: list[dict], k: int) -> list[dict]:
    return sorted(items, key=lambda x: x.get("missing_kw", 0.0), reverse=True)[: max(1, k)]


def diagnose_multi_house_power(*, loads_by_house, pv_by_house, bat_params_by_house, max_rows):
    """Procura violações instantâneas de potência, por casa e para o sistema.

    Devolve `(violacoes_de_sistema, violacoes_por_casa)`, cada uma limitada às
    `max_rows` piores.
    """
    houses = list(loads_by_house.keys())
    T = len(next(iter(loads_by_house.values())))
    sys_viol: list[dict] = []
    house_viol: list[dict] = []
    for t in range(T):
        total_load = total_pv = total_cap = 0.0
        for h in houses:
            load = float(loads_by_house[h][t])
            pv = float(pv_by_house[h][t])
            bat = bat_params_by_house.get(h) or {}
            # Aceita 'P_contracted' (novo) ou 'P_grid_max' (retrocompatibilidade)
            P_grid = float(bat.get("P_contracted", bat.get("P_grid_max", 0.0)))
            P_dis = float(bat.get("P_dis_max", 0.0))
            max_supply_h = pv + P_grid + P_dis
            missing_h = load - max_supply_h
            if missing_h > 1e-6:
                house_viol.append(dict(
                    house=h, t=t, load_kw=load, pv_kw=pv,
                    max_supply_kw=max_supply_h, missing_kw=missing_h,
                    note="House Load > PV + P_grid_max + P_dis_max"))
            total_load += load
            total_pv += pv
            total_cap += pv + P_grid + P_dis
        missing_sys = total_load - total_cap
        if missing_sys > 1e-6:
            sys_viol.append(dict(
                house="__SYSTEM__", t=t, load_kw=total_load, pv_kw=total_pv,
                max_supply_kw=total_cap, missing_kw=missing_sys,
                note="System Load > sum(PV + P_grid_max + P_dis_max)."))
    return _top_k(sys_viol, max_rows), _top_k(house_viol, max_rows)


def _format_violations_section(title: str, viols: list[dict]) -> list[str]:
    lines = [title]
    for v in viols:
        lines.append(
            f"- t={v['t']} | Load={v['load_kw']:.3f} | PV={v['pv_kw']:.3f} | "
            f"MaxSupply~{v['max_supply_kw']:.3f} | Missing={v['missing_kw']:.3f}"
        )
    lines.append("")
    lines.append("Restrições ativas: P_imp<=P_contracted, P_exp<=P_contracted, P_dis<=P_dis_max.")
    return lines


def write_infeas_report(*, out_dir, case_name, solver_used, termination_condition,
                        debug_cfg, lines, model=None, model_kind="unknown"):
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "debug_infeasibility.txt"
    header = [
        f"CASE: {case_name}", f"MODE: {model_kind}",
        f"SOLVER_USED: {solver_used}", f"TERMINATION: {termination_condition}", "",
        "This case is infeasible (or solver did not return a feasible solution).",
        "Causas típicas: P_imp<=P_contracted, P_exp<=P_contracted, P_dis<=P_dis_max, limites SoC.", "",
    ]
    report_path.write_text("\n".join(header + lines), encoding="utf-8")
    if bool(debug_cfg.get("write_lp", True)) and model is not None:
        try:
            lp_path = out_dir / "debug_model.lp"
            model.write(str(lp_path), io_options={"symbolic_solver_labels": True})
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug("Failed to write debug_model.lp: %s", e)
    return report_path


def handle_infeasible(*, case_out_dir, case_name, debug_cfg, loads_by_house,
                      pv_by_house, bat_params_by_house, solver_used, solver,
                      termination, model, original_exc, model_kind="unknown"):
    """Escreve o relatório de diagnóstico e DEVOLVE (não levanta) o RuntimeError."""
    sys_viol, house_viol = diagnose_multi_house_power(
        loads_by_house=loads_by_house, pv_by_house=pv_by_house,
        bat_params_by_house=bat_params_by_house, max_rows=int(debug_cfg["max_rows"]))
    lines: list[str] = []
    if original_exc is not None:
        lines.append(f"ORIGINAL ERROR: {type(original_exc).__name__}: {original_exc}")
        lines.append("")
    if sys_viol:
        lines.extend(_format_violations_section("== SYSTEM-LEVEL POWER VIOLATIONS ==", sys_viol))
    if house_viol:
        if lines:
            lines.append("")
        lines.extend(_format_violations_section("== PER-HOUSE VIOLATIONS ==", house_viol))
    if not sys_viol and not house_viol:
        lines.append("No instantaneous power-cap violation found.")
        lines.append("Infeasibility may be due to inter-temporal SoC/energy constraints.")
    report = write_infeas_report(
        out_dir=case_out_dir, case_name=case_name,
        solver_used=str(solver_used or solver),
        termination_condition=str(termination or ""),
        debug_cfg=debug_cfg, lines=lines, model=model, model_kind=model_kind)
    exc_hint = f" | {type(original_exc).__name__}: {original_exc}" if original_exc is not None else ""
    err = RuntimeError(f"Infeasible/unsolved case '{case_name}'. See: {report}{exc_hint}")
    err.__cause__ = original_exc
    return err

"""Camada de resolução: cadeia de solvers e diagnóstico de infeasibilidade.

Extraído do `scripts/run_case.py`, que fazia tudo num único ficheiro de ~1000
linhas. O comportamento é idêntico — foi movimentação de código, não reescrita.
"""
from .solver_chain import (
    ACCEPTABLE_TERMINATIONS,
    choose_solver_chain,
    configure_highs_file_log,
    configure_highs_logging,
    solve_model_safe,
    solver_available_local,
)
from .diagnostics import diagnose_multi_house_power, handle_infeasible, write_infeas_report

__all__ = [
    "ACCEPTABLE_TERMINATIONS",
    "choose_solver_chain",
    "configure_highs_file_log",
    "configure_highs_logging",
    "solve_model_safe",
    "solver_available_local",
    "diagnose_multi_house_power",
    "handle_infeasible",
    "write_infeas_report",
]

from __future__ import annotations

from typing import Any, Mapping, Optional

import pyomo.environ as pyo


class SolveFailed(RuntimeError):
    pass


def _configure_highs_logging(opt) -> None:
    """Stream solver output and write `highs.log` when the solver supports it."""
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


def solve_model(
    model: pyo.ConcreteModel,
    *,
    solver: str = "highs",
    options: Optional[Mapping[str, Any]] = None,
    tee: bool = False,
    load_solutions: bool = True,
):
    """
    Robust Pyomo solve wrapper.

    For APPSI solvers (e.g. appsi_highs) we can prevent the
    "no feasible solution so cannot load" crash by setting load_solutions=False.
    The solution is then loaded after we check the termination condition.
    """
    opt = pyo.SolverFactory(solver)
    if opt is None:
        raise SolveFailed(f"SolverFactory('{solver}') returned None.")

    if options:
        for k, v in options.items():
            opt.options[k] = v

    # APPSI: avoid crash on infeasible by disabling auto-load
    if hasattr(opt, "config"):
        try:
            opt.config.load_solutions = bool(load_solutions)
        except AttributeError:
            pass

    if tee and (solver or "").lower() in ("highs", "appsi_highs"):
        _configure_highs_logging(opt)

    try:
        results = opt.solve(model, tee=tee, load_solutions=bool(load_solutions))
    except TypeError:
        # Some solvers don't accept the load_solutions kwarg
        results = opt.solve(model, tee=tee)

    # If we didn't load solutions automatically, load only if solver says it's feasible
    if not load_solutions:
        tc = getattr(results.solver, "termination_condition", None)
        if tc in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
            model.solutions.load_from(results)

    return results

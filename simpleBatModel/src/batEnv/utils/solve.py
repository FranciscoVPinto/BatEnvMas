from __future__ import annotations

from typing import Optional, Mapping, Any
import pyomo.environ as pyo


class SolveFailed(RuntimeError):
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

    Key point: for APPSI solvers (e.g. appsi_highs) we can prevent the
    'no feasible solution so cannot load' crash by setting load_solutions=False.
    Then we decide whether to load solutions after checking termination_condition.
    """
    opt = pyo.SolverFactory(solver)
    if opt is None:
        raise SolveFailed(f"SolverFactory('{solver}') returned None.")

    # Apply options
    if options:
        for k, v in options.items():
            opt.options[k] = v

    # APPSI: avoid crash on infeasible by disabling auto-load
    if hasattr(opt, "config"):
        try:
            opt.config.load_solutions = bool(load_solutions)
        except Exception:
            pass

    # HiGHS logging: make tee actually stream output for appsi_highs and ensure HiGHS logs
    # When tee=True we also set sensible HiGHS verbosity defaults unless user already provided them.
    if tee and (solver or "").lower() in ("highs", "appsi_highs"):
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
        try:
            if hasattr(opt, "options"):
                opt.options.setdefault("output_flag", True)
                opt.options.setdefault("log_to_console", True)
                opt.options.setdefault("log_file", "highs.log")
                # More frequent MIP progress lines (seconds); harmless for LPs
                opt.options.setdefault("mip_min_logging_interval", 1)
        except Exception:
            pass

    try:
        results = opt.solve(model, tee=tee, load_solutions=bool(load_solutions))
    except TypeError:
        # Some solvers don’t accept load_solutions kwarg here
        results = opt.solve(model, tee=tee)

    # If we didn’t load solutions automatically, load only if solver says it’s feasible
    if not load_solutions:
        tc = getattr(results.solver, "termination_condition", None)
        if tc in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
            model.solutions.load_from(results)

    return results
from __future__ import annotations

from typing import Optional, Mapping, Any
import pyomo.environ as pyo


def solve_model(model: pyo.ConcreteModel, solver: str = "highs", options: Optional[Mapping[str, Any]] = None, tee: bool = False):
    opt = pyo.SolverFactory(solver)
    if opt is None:
        raise RuntimeError(f"SolverFactory('{solver}') returned None. Is the solver installed/available?")

    if options:
        for k, v in options.items():
            opt.options[k] = v

    results = opt.solve(model, tee=tee)
    return results

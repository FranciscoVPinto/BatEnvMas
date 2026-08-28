"""
Selecção e invocação do solver, com cadeia de fallback.

Cadeia: `appsi_highs` -> `highs` -> `cbc` -> `glpk`, usando o primeiro
disponível no ambiente. O solver preferido (vindo do runset) é tentado
primeiro.

Movido de `scripts/run_case.py` sem alterações de comportamento.
"""
from __future__ import annotations

import logging
import shutil

import pyomo.environ as pyo

logger = logging.getLogger("batEnv.solving")


# Terminações do solver aceites como "temos solução utilizável".
# Definida UMA vez (havia 4 cópias locais idênticas no run_case, um convite a
# que uma delas divergisse). `maxTimeLimit` só é aceite depois de confirmar que
# existe mesmo uma solução carregada — ver os pontos de uso.
ACCEPTABLE_TERMINATIONS = (
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.feasible,
    pyo.TerminationCondition.maxTimeLimit,
)


def solver_available_local(name: str) -> bool:
    """Verifica se um solver está instalado, sem instanciar o modelo."""
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
    except Exception:  # noqa: BLE001
        return False
    return bool(opt) and opt.available(exception_flag=False)


def choose_solver_chain(preferred: str | None) -> list[str]:
    chain = [preferred] if preferred else []
    for s in ("appsi_highs", "highs", "cbc", "glpk"):
        if s not in chain:
            chain.append(s)
    return chain


def configure_highs_logging(opt) -> None:
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


def configure_highs_file_log(opt, log_path) -> None:
    """Escreve o log nativo do HiGHS em `log_path`, independentemente do `tee`.

    NÃO ecoa para a consola (evita inundar o output do Spyder em corridas de
    horizonte deslizante com centenas de janelas) — o objectivo é ter sempre em
    disco o log da ÚLTIMA janela tentada, para que uma falha possa ser
    diagnosticada sem repetir a corrida com `tee=True`.
    """
    if hasattr(opt, "options"):
        try:
            opt.options["output_flag"] = True
            opt.options["log_to_console"] = False
            opt.options["log_file"] = str(log_path)
            opt.options.setdefault("mip_min_logging_interval", 1)
        except (AttributeError, TypeError):
            pass


def solve_model_safe(model, *, solver, options, tee, log_file=None):
    """Resolve `model` percorrendo a cadeia de solvers disponíveis.

    Devolve `(results, solver_name_usado)`. Levanta `RuntimeError` se nenhum
    solver da cadeia conseguir resolver.
    """
    chain = [s for s in choose_solver_chain(solver) if solver_available_local(s)]
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
                    # HiGHS espera doubles nas opções numéricas; coagir int -> float.
                    opt.options[k] = float(v) if isinstance(v, int) else v
            if s in ("appsi_highs", "highs"):
                if tee:
                    configure_highs_logging(opt)
                if log_file is not None:
                    # log_file explícito ganha ao genérico do tee — sempre
                    # sobrescrito, para reflectir o solve mais recente.
                    configure_highs_file_log(opt, log_file)
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
            if not auto_load and tc in ACCEPTABLE_TERMINATIONS:
                try:
                    model.solutions.load_from(results)
                except Exception as load_err:  # noqa: BLE001
                    logger.warning("Could not load solution (tc=%s): %s", tc, load_err)
            return results, s
        except Exception as e:  # noqa: BLE001
            logger.debug("Solver %s failed: %s", s, e)
            last_err = e
    raise RuntimeError(f"Failed to solve model. Last error: {last_err}")

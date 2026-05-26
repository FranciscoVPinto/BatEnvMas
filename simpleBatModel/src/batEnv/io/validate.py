from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Tuple

from .case_schema import CASE_SCHEMA
from .orchestration_schemas import PLOTSET_SCHEMA, RUNSET_SCHEMA

logger = logging.getLogger(__name__)


def _keys_to_str(d: dict[Any, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in d.items()}


def canonicalize_case_cfg(cfg: dict[str, Any]) -> Tuple[dict[str, Any], list[str]]:
    """
    Returns (cfg2, warnings). Normalises dict keys to str in houses,
    data.loads, sharing.alpha, and sharing.alpha_profile.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Case cfg must be a dict")

    c = deepcopy(cfg)
    warnings: list[str] = []

    houses = c.get("houses", {})
    if isinstance(houses, dict):
        c["houses"] = _keys_to_str(houses)
    else:
        warnings.append("houses is not a dict (cannot canonicalize keys)")

    data = c.get("data", {})
    if isinstance(data, dict):
        loads = data.get("loads", {})
        if isinstance(loads, dict):
            data["loads"] = _keys_to_str(loads)
        else:
            warnings.append("data.loads is not a dict (cannot canonicalize keys)")
        c["data"] = data

    sharing = c.get("sharing", {})
    if isinstance(sharing, dict):
        alpha = sharing.get("alpha", None)
        if isinstance(alpha, dict):
            sharing["alpha"] = _keys_to_str(alpha)

        alpha_prof = sharing.get("alpha_profile", None)
        if isinstance(alpha_prof, dict):
            sharing["alpha_profile"] = _keys_to_str(alpha_prof)

        c["sharing"] = sharing

    return c, warnings


def validate_case_cfg_basic(cfg: dict[str, Any]) -> None:
    """Validacao estrutural rapida. Falha cedo com mensagens claras."""
    if not isinstance(cfg.get("time", {}), dict):
        raise ValueError("time must be a dict")
    if not isinstance(cfg.get("data", {}), dict):
        raise ValueError("data must be a dict")
    if not isinstance(cfg.get("houses", {}), dict) or not cfg.get("houses"):
        raise ValueError("houses must be a non-empty dict")

    loads = cfg["data"].get("loads", {})
    if not isinstance(loads, dict) or not loads:
        raise ValueError("data.loads must be a non-empty dict")


def _validate_against(cfg: dict[str, Any], schema: dict, *, label: str) -> None:
    """Shared schema-check helper. No-op + warning when jsonschema is missing."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        logger.warning(
            "jsonschema not installed; skipping %s schema validation. "
            "Install with `pip install jsonschema` to enable it.",
            label,
        )
        return

    try:
        jsonschema.validate(instance=cfg, schema=schema)
    except jsonschema.ValidationError as e:
        # YAML-friendly path like "houses.Apt1.battery.E_max"
        loc = ".".join(str(p) for p in e.absolute_path) or "<root>"
        raise ValueError(f"{label} cfg invalid at '{loc}': {e.message}") from e


def validate_case_cfg_schema(cfg: dict[str, Any]) -> None:
    """Full JSON-schema validation of a case cfg."""
    _validate_against(cfg, CASE_SCHEMA, label="Case")


def validate_runset_cfg(cfg: dict[str, Any]) -> None:
    """Validate a runset / parent-runset / single-experiment YAML."""
    _validate_against(cfg, RUNSET_SCHEMA, label="Runset")


def validate_plotset_cfg(cfg: dict[str, Any]) -> None:
    """Validate a plotset / parent-plotset / single-experiment-plot YAML."""
    _validate_against(cfg, PLOTSET_SCHEMA, label="Plotset")

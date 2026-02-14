from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple


def _keys_to_str(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in d.items()}


def canonicalize_case_cfg(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (cfg2, warnings)

    Canonicaliza IDs (ex.: 1 vs "1") para STR em:
      - houses
      - data.loads
      - sharing.alpha
      - sharing.alpha_profile

    Não altera valores numéricos.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Case cfg must be a dict")

    c = deepcopy(cfg)
    warnings: List[str] = []

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


def validate_case_cfg_basic(cfg: Dict[str, Any]) -> None:
    """
    Validação estrutural (rápida). Falha cedo com mensagens claras.
    """
    if not isinstance(cfg.get("time", {}), dict):
        raise ValueError("time must be a dict")
    if not isinstance(cfg.get("data", {}), dict):
        raise ValueError("data must be a dict")
    if not isinstance(cfg.get("houses", {}), dict) or not cfg.get("houses"):
        raise ValueError("houses must be a non-empty dict")

    loads = cfg["data"].get("loads", {})
    if not isinstance(loads, dict) or not loads:
        raise ValueError("data.loads must be a non-empty dict")

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .loaders import load_series_csv_1col


def _abs_from_root(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def _normalize(alpha: Dict[str, float]) -> Dict[str, float]:
    s = sum(alpha.values())
    if s <= 0:
        raise ValueError("Cannot normalize alphas: sum <= 0")
    return {k: v / s for k, v in alpha.items()}


def _equal_split(houses: List[str]) -> Dict[str, float]:
    if not houses:
        raise ValueError("No houses provided")
    w = 1.0 / len(houses)
    return {h: w for h in houses}


def _alphas_from_consumption_instant(
    loads_by_house: Dict[str, List[float]],
    houses: List[str],
    T: int,
    *,
    zero_denom: str = "equal_split",  # equal_split | keep_previous
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    alpha_i(t) = Load_i(t) / sum_j Load_j(t)

    Se sum_j Load_j(t) == 0:
      - zero_denom="equal_split": alpha_i(t) = 1/N
      - zero_denom="keep_previous": alpha_i(t) = alpha_i(t-1) (t=0 -> equal_split)
    """
    alpha_series: Dict[str, List[float]] = {h: [0.0] * T for h in houses}
    denom_series: List[float] = []

    prev = _equal_split(houses)

    for t in range(T):
        denom = float(sum(float(loads_by_house[h][t]) for h in houses))
        denom_series.append(denom)

        if denom > 0:
            a_t = {h: float(loads_by_house[h][t]) / denom for h in houses}
            prev = a_t
        else:
            if zero_denom == "equal_split":
                a_t = _equal_split(houses)
                prev = a_t
            elif zero_denom == "keep_previous":
                a_t = prev
            else:
                raise ValueError(f"Unsupported fallback.zero_denom='{zero_denom}'")

        for h in houses:
            alpha_series[h][t] = float(a_t[h])

    info = {
        "fallback_mode": "consumption_instant",
        "zero_denom": zero_denom,
        "denom_min": float(min(denom_series)) if denom_series else None,
        "denom_max": float(max(denom_series)) if denom_series else None,
    }
    return alpha_series, info


def _alphas_from_consumption_mean(
    loads_by_house: Dict[str, List[float]],
    houses: List[str],
    T: int,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    alpha_i = mean(Load_i) / sum_j mean(Load_j)  (constante no tempo)
    """
    means = {h: float(sum(loads_by_house[h])) / float(T) for h in houses}
    denom = float(sum(means.values()))
    if denom > 0:
        alpha_const = {h: means[h] / denom for h in houses}
    else:
        alpha_const = _equal_split(houses)

    alpha_series = {h: [float(alpha_const[h])] * T for h in houses}
    info = {"fallback_mode": "consumption_mean", "means": means, "mean_sum": denom}
    return alpha_series, info


def prepare_pv_by_house(
    cfg: dict,
    *,
    houses: List[str],
    T: int,
    root: Path,
    loads_by_house: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Any], Dict[str, Any]]:
    """
    Constrói PV_i(t) por casa.

    Returns:
      pv_by_house: {house_id: [PV_i(t)]}
      info: diagnóstico pequeno (ok para meta.yaml)
      debug: séries (ideal exportar para CSV em results/<case>/preprocess/)

    Modos PV (mutuamente exclusivos):
      A) data.pv (legacy): str ou dict por casa
      B) data.pv_total + sharing (shared PV)

    Shared PV suporta:
      - sharing.alpha (escalar por casa)
      - sharing.alpha_profile (série por casa)
      - normalize / strict_sum_to_one
      - fallback:
          mode: none | consumption_instant | consumption_mean
          apply_when: invalid_or_missing | always
          zero_denom: equal_split | keep_previous   (apenas no instant)
    """
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("data must be a dict")

    info: Dict[str, Any] = {"warnings": []}
    debug: Dict[str, Any] = {}

    has_pv = "pv" in data_cfg
    has_pv_total = "pv_total" in data_cfg

    if has_pv and has_pv_total:
        raise ValueError("Ambiguous PV definition: use either data.pv OR data.pv_total, not both.")
    if (not has_pv) and (not has_pv_total):
        raise ValueError("Missing PV definition: provide either data.pv or data.pv_total.")

    # -----------------------
    # Mode A: per-house PV
    # -----------------------
    if has_pv:
        pv_cfg = data_cfg.get("pv")
        info["mode"] = "per_house"
        pv_by_house: Dict[str, List[float]] = {}

        if isinstance(pv_cfg, str):
            pv_common = load_series_csv_1col(_abs_from_root(root, pv_cfg), T=T)
            for h in houses:
                pv_by_house[h] = list(pv_common)

        elif isinstance(pv_cfg, dict):
            pv_cfg_str = {str(k): v for k, v in pv_cfg.items()}
            missing = [h for h in houses if h not in pv_cfg_str]
            if missing:
                raise ValueError(f"Missing data.pv for houses: {missing}")
            for h in houses:
                pv_by_house[h] = load_series_csv_1col(_abs_from_root(root, pv_cfg_str[h]), T=T)

        else:
            raise ValueError("data.pv must be a string path or a dict {house: path}")

        debug["pv_mode"] = "per_house"
        return pv_by_house, info, debug

    # -----------------------
    # Mode B: shared PV_total
    # -----------------------
    pv_total_path = data_cfg.get("pv_total")
    pv_total = load_series_csv_1col(_abs_from_root(root, pv_total_path), T=T)

    sharing_cfg = cfg.get("sharing", {})
    if not isinstance(sharing_cfg, dict):
        raise ValueError("sharing must be a dict when using data.pv_total")

    mode = sharing_cfg.get("mode", "fixed_alpha")
    if mode != "fixed_alpha":
        raise ValueError(f"Unsupported sharing.mode='{mode}' (supported: fixed_alpha)")

    normalize = bool(sharing_cfg.get("normalize", False))
    strict_sum_to_one = bool(sharing_cfg.get("strict_sum_to_one", False))

    fb_cfg = sharing_cfg.get("fallback", {}) if isinstance(sharing_cfg.get("fallback", {}), dict) else {}
    fb_mode = str(fb_cfg.get("mode", "none"))
    fb_apply_when = str(fb_cfg.get("apply_when", "invalid_or_missing"))  # or "always"
    fb_zero_denom = str(fb_cfg.get("zero_denom", "equal_split"))

    info.update(
        {
            "mode": "shared_alpha",
            "normalize": normalize,
            "strict_sum_to_one": strict_sum_to_one,
            "fallback": {"mode": fb_mode, "apply_when": fb_apply_when, "zero_denom": fb_zero_denom},
        }
    )

    alpha_map = sharing_cfg.get("alpha", None)
    alpha_profile = sharing_cfg.get("alpha_profile", None)

    if (alpha_map is not None) and (alpha_profile is not None):
        raise ValueError("Provide only one of sharing.alpha or sharing.alpha_profile, not both.")

    def _need_fallback(reason: str) -> bool:
        info["warnings"].append(f"FALLBACK trigger: {reason}")
        return fb_mode in ("consumption_instant", "consumption_mean")

    def _build_alpha_series_via_fallback() -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        if loads_by_house is None:
            raise ValueError("Fallback based on consumption requires loads_by_house to be provided.")
        if fb_mode == "consumption_instant":
            return _alphas_from_consumption_instant(loads_by_house, houses, T, zero_denom=fb_zero_denom)
        if fb_mode == "consumption_mean":
            return _alphas_from_consumption_mean(loads_by_house, houses, T)
        raise ValueError(f"Unsupported fallback.mode='{fb_mode}'")

    # Fallback always
    if fb_apply_when == "always":
        if fb_mode == "none":
            raise ValueError("fallback.apply_when=always requires fallback.mode != none")
        alpha_series, fb_info = _build_alpha_series_via_fallback()
        info["fallback_used"] = True
        info["fallback_details"] = fb_info

        alpha_sum = [sum(alpha_series[h][t] for h in houses) for t in range(T)]
        pv_by_house = {h: [alpha_series[h][t] * pv_total[t] for t in range(T)] for h in houses}

        debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
        return pv_by_house, info, debug

    # Try provided alpha definitions, fallback only if invalid/missing
    if alpha_map is None and alpha_profile is None:
        if fb_mode == "none":
            raise ValueError("For shared PV, provide sharing.alpha or sharing.alpha_profile, or set a fallback.mode.")
        alpha_series, fb_info = _build_alpha_series_via_fallback()
        info["fallback_used"] = True
        info["fallback_details"] = fb_info

        alpha_sum = [sum(alpha_series[h][t] for h in houses) for t in range(T)]
        pv_by_house = {h: [alpha_series[h][t] * pv_total[t] for t in range(T)] for h in houses}

        debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
        return pv_by_house, info, debug

    # --- scalar alpha
    if alpha_map is not None:
        if not isinstance(alpha_map, dict):
            raise ValueError("sharing.alpha must be a dict {house: scalar}")

        alpha_map_str = {str(k): float(v) for k, v in alpha_map.items()}

        missing = [h for h in houses if h not in alpha_map_str]
        if missing:
            if fb_mode != "none" and _need_fallback(f"missing scalar alpha for {missing}"):
                alpha_series, fb_info = _build_alpha_series_via_fallback()
                info["fallback_used"] = True
                info["fallback_details"] = fb_info
                alpha_sum = [sum(alpha_series[h][t] for h in houses) for t in range(T)]
                pv_by_house = {h: [alpha_series[h][t] * pv_total[t] for t in range(T)] for h in houses}
                debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
                return pv_by_house, info, debug
            raise ValueError(f"Missing sharing.alpha for houses: {missing}")

        for h in houses:
            if alpha_map_str[h] < 0:
                if fb_mode != "none" and _need_fallback(f"negative scalar alpha for house {h}"):
                    alpha_series, fb_info = _build_alpha_series_via_fallback()
                    info["fallback_used"] = True
                    info["fallback_details"] = fb_info
                    alpha_sum = [sum(alpha_series[hh][t] for hh in houses) for t in range(T)]
                    pv_by_house = {hh: [alpha_series[hh][t] * pv_total[t] for t in range(T)] for hh in houses}
                    debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
                    return pv_by_house, info, debug
                raise ValueError(f"Alpha for house '{h}' is negative: {alpha_map_str[h]}")

        s = sum(alpha_map_str[h] for h in houses)
        info["alpha_sum"] = float(s)

        if strict_sum_to_one and abs(s - 1.0) > 1e-6:
            if fb_mode != "none" and _need_fallback(f"strict_sum_to_one violated (sum={s})"):
                alpha_series, fb_info = _build_alpha_series_via_fallback()
                info["fallback_used"] = True
                info["fallback_details"] = fb_info
                alpha_sum = [sum(alpha_series[h][t] for h in houses) for t in range(T)]
                pv_by_house = {h: [alpha_series[h][t] * pv_total[t] for t in range(T)] for h in houses}
                debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
                return pv_by_house, info, debug
            raise ValueError(f"Sum of sharing.alpha is {s} but strict_sum_to_one=true (expected 1.0).")

        alpha_used_scalar = dict(alpha_map_str)
        if normalize and abs(s - 1.0) > 1e-12:
            alpha_used_scalar = _normalize(alpha_used_scalar)
            info["alpha_sum_normalized"] = float(sum(alpha_used_scalar[h] for h in houses))
        elif abs(s - 1.0) > 1e-6:
            info["warnings"].append(f"Sum of alphas is {s:.6f} (not 1.0). PV allocations follow given weights.")

        alpha_series = {h: [float(alpha_used_scalar[h])] * T for h in houses}
        alpha_sum = [sum(alpha_series[h][t] for h in houses) for t in range(T)]
        pv_by_house = {h: [alpha_series[h][t] * pv_total[t] for t in range(T)] for h in houses}

        debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
        return pv_by_house, info, debug

    # --- alpha_profile
    if not isinstance(alpha_profile, dict):
        raise ValueError("sharing.alpha_profile must be a dict {house: csv_path}")

    alpha_profile_str = {str(k): v for k, v in alpha_profile.items()}
    missing = [h for h in houses if h not in alpha_profile_str]
    if missing:
        if fb_mode != "none" and _need_fallback(f"missing alpha_profile for {missing}"):
            alpha_series, fb_info = _build_alpha_series_via_fallback()
            info["fallback_used"] = True
            info["fallback_details"] = fb_info
            alpha_sum = [sum(alpha_series[h][t] for h in houses) for t in range(T)]
            pv_by_house = {h: [alpha_series[h][t] * pv_total[t] for t in range(T)] for h in houses}
            debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
            return pv_by_house, info, debug
        raise ValueError(f"Missing sharing.alpha_profile for houses: {missing}")

    alpha_series_by_house: Dict[str, List[float]] = {}
    for h in houses:
        a_ser = load_series_csv_1col(_abs_from_root(root, alpha_profile_str[h]), T=T)
        for t, val in enumerate(a_ser):
            if float(val) < 0:
                if fb_mode != "none" and _need_fallback(f"negative alpha_profile for house {h} at t={t}"):
                    alpha_series, fb_info = _build_alpha_series_via_fallback()
                    info["fallback_used"] = True
                    info["fallback_details"] = fb_info
                    alpha_sum = [sum(alpha_series[hh][tt] for hh in houses) for tt in range(T)]
                    pv_by_house2 = {hh: [alpha_series[hh][tt] * pv_total[tt] for tt in range(T)] for hh in houses}
                    debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
                    return pv_by_house2, info, debug
                raise ValueError(f"Alpha_profile negative for house '{h}' at t={t}: {val}")
        alpha_series_by_house[h] = [float(x) for x in a_ser]

    pv_by_house: Dict[str, List[float]] = {h: [0.0] * T for h in houses}
    alpha_sum_series: List[float] = []
    alpha_used_series: Dict[str, List[float]] = {h: [0.0] * T for h in houses}

    for t in range(T):
        a_t = {h: alpha_series_by_house[h][t] for h in houses}
        s = float(sum(a_t.values()))
        alpha_sum_series.append(s)

        if strict_sum_to_one and abs(s - 1.0) > 1e-6:
            if fb_mode != "none" and _need_fallback(f"strict_sum_to_one violated at t={t} (sum={s})"):
                alpha_series, fb_info = _build_alpha_series_via_fallback()
                info["fallback_used"] = True
                info["fallback_details"] = fb_info
                alpha_sum = [sum(alpha_series[hh][tt] for hh in houses) for tt in range(T)]
                pv_by_house2 = {hh: [alpha_series[hh][tt] * pv_total[tt] for tt in range(T)] for hh in houses}
                debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_series, "alpha_sum": alpha_sum})
                return pv_by_house2, info, debug
            raise ValueError(f"At t={t}, sum(alpha_profile)={s} but strict_sum_to_one=true.")

        if normalize and s > 0 and abs(s - 1.0) > 1e-12:
            a_t = _normalize(a_t)

        for h in houses:
            alpha_used_series[h][t] = float(a_t[h])
            pv_by_house[h][t] = float(a_t[h]) * float(pv_total[t])

    info["alpha_sum_min"] = float(min(alpha_sum_series)) if alpha_sum_series else None
    info["alpha_sum_max"] = float(max(alpha_sum_series)) if alpha_sum_series else None

    if (not normalize) and alpha_sum_series:
        if abs(info["alpha_sum_min"] - 1.0) > 1e-6 or abs(info["alpha_sum_max"] - 1.0) > 1e-6:
            info["warnings"].append(
                f"alpha_profile sums vary (min={info['alpha_sum_min']:.6f}, max={info['alpha_sum_max']:.6f}) and normalize=false."
            )

    debug.update({"pv_mode": "shared_alpha", "pv_total": pv_total, "alpha_used": alpha_used_series, "alpha_sum": alpha_sum_series})
    return pv_by_house, info, debug

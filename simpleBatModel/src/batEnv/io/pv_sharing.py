from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .loaders import load_series_csv_1col


def _abs_from_root(root: Path, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def _normalize(alpha: Dict[str, float]) -> Dict[str, float]:
    s = sum(alpha.values())
    if s <= 0:
        raise ValueError("Cannot normalize alphas: sum <= 0")
    return {k: v / s for k, v in alpha.items()}


def _equal_split(houses: List[str]) -> Dict[str, float]:
    w = 1.0 / len(houses)
    return {h: w for h in houses}


# --- Modos de calculo de alpha ---

def _alphas_from_consumption_instant(
    loads_by_house: Dict[str, List[float]],
    houses: List[str],
    T: int,
    *,
    zero_denom: str = "equal_split",
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """alpha[h,t] = Load[h,t] / sum_h Load[h,t].  Garante sum=1 por construcao."""
    alpha_series: Dict[str, List[float]] = {h: [0.0] * T for h in houses}
    denom_series: List[float] = []
    prev = _equal_split(houses)

    for t in range(T):
        denom = float(sum(float(loads_by_house[h][t]) for h in houses))
        denom_series.append(denom)
        if denom > 0:
            a_t = {h: float(loads_by_house[h][t]) / denom for h in houses}
            prev = a_t
        elif zero_denom == "equal_split":
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
    """alpha[h] = mean(Load[h]) / sum_h mean(Load[h]).  Constante no tempo."""
    means = {h: float(sum(loads_by_house[h])) / T for h in houses}
    denom = float(sum(means.values()))
    alpha_const = {h: means[h] / denom for h in houses} if denom > 0 else _equal_split(houses)
    alpha_series = {h: [float(alpha_const[h])] * T for h in houses}
    info = {"fallback_mode": "consumption_mean", "means": means, "mean_sum": denom}
    return alpha_series, info


# --- Carregamento de PV ---

def _per_house_pv(data_cfg: dict, houses: List[str], T: int, root: Path) -> Dict[str, List[float]]:
    pv_cfg = data_cfg.get("pv")
    if isinstance(pv_cfg, str):
        pv_common = load_series_csv_1col(_abs_from_root(root, pv_cfg), T=T)
        return {h: list(pv_common) for h in houses}
    if isinstance(pv_cfg, dict):
        pv_cfg_str = {str(k): v for k, v in pv_cfg.items()}
        missing = [h for h in houses if h not in pv_cfg_str]
        if missing:
            raise ValueError(f"Missing data.pv for houses: {missing}")
        return {h: load_series_csv_1col(_abs_from_root(root, pv_cfg_str[h]), T=T) for h in houses}
    raise ValueError("data.pv must be a string path or a dict {house: path}")


def load_pv_total(data_cfg: dict, root: Path, T: int) -> List[float]:
    """Carrega a serie de PV total da comunidade (soma de todos os inversores)."""
    return _load_pv_total(data_cfg, root, T)


def _load_pv_total(data_cfg: dict, root: Path, T: int) -> List[float]:
    pv_total_path = data_cfg.get("pv_total")
    if isinstance(pv_total_path, (list, tuple)):
        pv_total = [0.0] * T
        for p in pv_total_path:
            series = load_series_csv_1col(_abs_from_root(root, p), T=T)
            pv_total = [a + b for a, b in zip(pv_total, series)]
        return pv_total
    return load_series_csv_1col(_abs_from_root(root, pv_total_path), T=T)


# --- Validacao e construcao de alpha a partir do YAML ---

def _alphas_from_scalar(
    alpha_map: Any,
    houses: List[str],
    T: int,
    *,
    normalize: bool,
    strict_sum_to_one: bool,
    info: Dict[str, Any],
    fallback_or_raise: Callable,
) -> Dict[str, List[float]]:
    if not isinstance(alpha_map, dict):
        raise ValueError("sharing.alpha must be a dict {house: scalar}")

    alpha_map_str = {str(k): float(v) for k, v in alpha_map.items()}
    missing = [h for h in houses if h not in alpha_map_str]
    if missing:
        fb = fallback_or_raise(f"missing scalar alpha for {missing}",
                               f"Missing sharing.alpha for houses: {missing}")
        if fb is not None:
            return fb

    for h in houses:
        if alpha_map_str[h] < 0:
            fb = fallback_or_raise(f"negative scalar alpha for house {h}",
                                   f"Alpha for house '{h}' is negative: {alpha_map_str[h]}")
            if fb is not None:
                return fb

    s = sum(alpha_map_str[h] for h in houses)
    info["alpha_sum"] = float(s)

    if strict_sum_to_one and abs(s - 1.0) > 1e-6:
        fb = fallback_or_raise(f"strict_sum_to_one violated (sum={s})",
                               f"Sum of sharing.alpha is {s} but strict_sum_to_one=true.")
        if fb is not None:
            return fb

    alpha_used = dict(alpha_map_str)
    if normalize and abs(s - 1.0) > 1e-12:
        alpha_used = _normalize(alpha_used)
        info["alpha_sum_normalized"] = float(sum(alpha_used[h] for h in houses))
    elif abs(s - 1.0) > 1e-6:
        info["warnings"].append(f"Sum of alphas is {s:.6f} (not 1.0).")

    return {h: [float(alpha_used[h])] * T for h in houses}


def _alphas_from_profile(
    alpha_profile: Any,
    houses: List[str],
    T: int,
    root: Path,
    *,
    normalize: bool,
    strict_sum_to_one: bool,
    info: Dict[str, Any],
    fallback_or_raise: Callable,
) -> Tuple[Dict[str, List[float]], Optional[List[float]]]:
    if not isinstance(alpha_profile, dict):
        raise ValueError("sharing.alpha_profile must be a dict {house: csv_path}")

    alpha_profile_str = {str(k): v for k, v in alpha_profile.items()}
    missing = [h for h in houses if h not in alpha_profile_str]
    if missing:
        fb = fallback_or_raise(f"missing alpha_profile for {missing}",
                               f"Missing sharing.alpha_profile for houses: {missing}")
        if fb is not None:
            return fb, None

    raw: Dict[str, List[float]] = {}
    for h in houses:
        a_ser = load_series_csv_1col(_abs_from_root(root, alpha_profile_str[h]), T=T)
        for t, val in enumerate(a_ser):
            if float(val) < 0:
                fb = fallback_or_raise(f"negative alpha_profile for house {h} at t={t}",
                                       f"Alpha_profile negative for '{h}' at t={t}: {val}")
                if fb is not None:
                    return fb, None
        raw[h] = [float(x) for x in a_ser]

    out: Dict[str, List[float]] = {h: [0.0] * T for h in houses}
    alpha_sum_series: List[float] = []

    for t in range(T):
        a_t = {h: raw[h][t] for h in houses}
        s = float(sum(a_t.values()))
        alpha_sum_series.append(s)
        if strict_sum_to_one and abs(s - 1.0) > 1e-6:
            fb = fallback_or_raise(f"strict_sum_to_one violated at t={t} (sum={s})",
                                   f"At t={t}, sum(alpha_profile)={s} != 1.")
            if fb is not None:
                return fb, None
        if normalize and s > 0 and abs(s - 1.0) > 1e-12:
            a_t = _normalize(a_t)
        for h in houses:
            out[h][t] = float(a_t[h])

    info["alpha_sum_min"] = float(min(alpha_sum_series)) if alpha_sum_series else None
    info["alpha_sum_max"] = float(max(alpha_sum_series)) if alpha_sum_series else None
    if (not normalize) and alpha_sum_series and (
        abs(info["alpha_sum_min"] - 1.0) > 1e-6 or abs(info["alpha_sum_max"] - 1.0) > 1e-6
    ):
        info["warnings"].append(
            f"alpha_profile sums vary (min={info['alpha_sum_min']:.6f}, "
            f"max={info['alpha_sum_max']:.6f}) and normalize=false."
        )
    return out, alpha_sum_series


# --- Ponto central: converte alphas em PV por casa ---

def _finalize_shared(
    alpha_series: Dict[str, List[float]],
    pv_total: List[float],
    houses: List[str],
    T: int,
    info: Dict[str, Any],
    debug: Dict[str, Any],
    *,
    alpha_sum: Optional[List[float]] = None,
    tol: float = 1e-6,
) -> Tuple[Dict[str, List[float]], Dict[str, Any], Dict[str, Any]]:
    """
    Garante sum_h alpha[h,t] = 1 para todo t e calcula PV[h,t] = alpha[h,t] * PV_total[t].

    Se a soma diferir de 1 (alem de tol), normaliza automaticamente e regista aviso.
    """
    if alpha_sum is None:
        alpha_sum = [sum(alpha_series[h][t] for h in houses) for t in range(T)]

    needs_norm = any(abs(s - 1.0) > tol for s in alpha_sum)
    if needs_norm:
        bad_t = [(t, alpha_sum[t]) for t in range(T) if abs(alpha_sum[t] - 1.0) > tol]
        info["warnings"].append(
            f"sum(alpha) != 1 em {len(bad_t)} instante(s) "
            f"(ex.: t={bad_t[0][0]}, soma={bad_t[0][1]:.6f}). A normalizar."
        )
        for t in range(T):
            s = alpha_sum[t]
            if s <= 0:
                equal = 1.0 / max(1, len(houses))
                for h in houses:
                    alpha_series[h][t] = equal
                alpha_sum[t] = 1.0
            elif abs(s - 1.0) > tol:
                for h in houses:
                    alpha_series[h][t] /= s
                alpha_sum[t] = 1.0

    # Verificacao de sanidade pos-normalizacao
    for t in range(T):
        s = sum(alpha_series[h][t] for h in houses)
        if abs(s - 1.0) > tol:
            raise ValueError(
                f"sum(alpha[h, t={t}]) = {s:.8f} != 1 mesmo apos normalizacao. "
                "Verifique se os alphas sao todos nao-negativos."
            )

    pv_by_house = {
        h: [alpha_series[h][t] * pv_total[t] for t in range(T)]
        for h in houses
    }
    debug.update({
        "pv_mode": "shared_alpha",
        "pv_total": pv_total,
        "alpha_used": alpha_series,
        "alpha_sum": alpha_sum,
        "alpha_sum_normalized": needs_norm,
    })
    return pv_by_house, info, debug


# --- API publica ---

def prepare_pv_by_house(
    cfg: dict,
    *,
    houses: List[str],
    T: int,
    root: Path,
    loads_by_house: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Any], Dict[str, Any]]:
    """Constroi PV_i(t) por casa a partir da configuracao do caso YAML."""
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("data must be a dict")

    info: Dict[str, Any] = {"warnings": []}
    debug: Dict[str, Any] = {}

    # data.pv_total: null (herdado do base e anulado no filho) é tratado como ausente
    has_pv       = data_cfg.get("pv")       is not None
    has_pv_total = data_cfg.get("pv_total") is not None

    if has_pv and has_pv_total:
        raise ValueError("Ambiguous PV definition: use either data.pv OR data.pv_total, not both.")
    if not has_pv and not has_pv_total:
        raise ValueError("Missing PV definition: provide either data.pv or data.pv_total.")

    # Modo A: PV pre-calculado por casa
    if has_pv:
        info["mode"] = "per_house"
        debug["pv_mode"] = "per_house"
        return _per_house_pv(data_cfg, houses, T, root), info, debug

    # Modo B: PV total partilhado com alphas
    pv_total = _load_pv_total(data_cfg, root, T)

    sharing_cfg = cfg.get("sharing", {})
    if not isinstance(sharing_cfg, dict):
        raise ValueError("sharing must be a dict when using data.pv_total")

    mode = sharing_cfg.get("mode", "fixed_alpha")
    if mode != "fixed_alpha":
        raise ValueError(f"Unsupported sharing.mode='{mode}' (supported: fixed_alpha)")

    # normalize=True por defeito: garante sum(alpha)=1 mesmo com pesos aproximados no YAML
    normalize         = bool(sharing_cfg.get("normalize", True))
    strict_sum_to_one = bool(sharing_cfg.get("strict_sum_to_one", False))

    fb_cfg        = sharing_cfg.get("fallback", {}) or {}
    fb_mode       = str(fb_cfg.get("mode", "none"))
    fb_apply_when = str(fb_cfg.get("apply_when", "invalid_or_missing"))
    fb_zero_denom = str(fb_cfg.get("zero_denom", "equal_split"))

    info.update({
        "mode": "shared_alpha",
        "normalize": normalize,
        "strict_sum_to_one": strict_sum_to_one,
        "fallback": {"mode": fb_mode, "apply_when": fb_apply_when, "zero_denom": fb_zero_denom},
    })

    alpha_map     = sharing_cfg.get("alpha", None)
    alpha_profile = sharing_cfg.get("alpha_profile", None)

    if (alpha_map is not None) and (alpha_profile is not None):
        raise ValueError("Provide only one of sharing.alpha or sharing.alpha_profile, not both.")

    def build_fallback_alpha():
        if loads_by_house is None:
            raise ValueError("Fallback based on consumption requires loads_by_house.")
        if fb_mode == "consumption_instant":
            alpha_series, fb_info = _alphas_from_consumption_instant(
                loads_by_house, houses, T, zero_denom=fb_zero_denom)
        elif fb_mode == "consumption_mean":
            alpha_series, fb_info = _alphas_from_consumption_mean(loads_by_house, houses, T)
        else:
            raise ValueError(f"Unsupported fallback.mode='{fb_mode}'")
        info["fallback_used"] = True
        info["fallback_details"] = fb_info
        return alpha_series

    def fallback_or_raise(reason: str, raise_msg: str):
        if fb_mode == "none":
            raise ValueError(raise_msg)
        info["warnings"].append(f"FALLBACK trigger: {reason}")
        if fb_mode not in ("consumption_instant", "consumption_mean"):
            raise ValueError(raise_msg)
        return build_fallback_alpha()

    if fb_apply_when == "always":
        if fb_mode == "none":
            raise ValueError("fallback.apply_when=always requires fallback.mode != none")
        return _finalize_shared(build_fallback_alpha(), pv_total, houses, T, info, debug)

    if alpha_map is None and alpha_profile is None:
        if fb_mode == "none":
            raise ValueError(
                "For shared PV, provide sharing.alpha or sharing.alpha_profile, "
                "or set a fallback.mode.")
        return _finalize_shared(build_fallback_alpha(), pv_total, houses, T, info, debug)

    if alpha_map is not None:
        alpha_series = _alphas_from_scalar(
            alpha_map, houses, T,
            normalize=normalize, strict_sum_to_one=strict_sum_to_one,
            info=info, fallback_or_raise=fallback_or_raise)
        return _finalize_shared(alpha_series, pv_total, houses, T, info, debug)

    alpha_series, alpha_sum_pre = _alphas_from_profile(
        alpha_profile, houses, T, root,
        normalize=normalize, strict_sum_to_one=strict_sum_to_one,
        info=info, fallback_or_raise=fallback_or_raise)
    return _finalize_shared(alpha_series, pv_total, houses, T, info, debug,
                            alpha_sum=alpha_sum_pre)

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import datetime as dt

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge: dicts merge key-wise, scalars from `override` win.

    An explicit `null` in `override` DELETES the key from the merged result
    (rather than setting it to None) — this is how a child case cancels a
    key it inherited from a base via `extends:` (e.g. `data.pv_total: ~` to
    switch from shared PV to per-house PV without leaving a dangling
    `pv_total: null` alongside `pv:` in the merged config).
    """
    out = dict(base or {})
    for k, v in (override or {}).items():
        if v is None and k in out:
            del out[k]
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_case_yaml(path: str | Path, *, _seen: Optional[set] = None) -> dict:
    """
    Load a case YAML, with optional `extends:` inheritance.

    A YAML may declare:
        extends: relative/path/to/base.yaml          # single base
        extends: [base1.yaml, base2.yaml, ...]       # multiple, later wins

    Bases are merged recursively (deep_merge); the current YAML's keys then
    override the merged base. The `extends` key itself is removed from the
    returned dict. Cyclic chains raise ValueError.
    """
    p = Path(path).resolve()

    seen = set(_seen) if _seen else set()
    if p in seen:
        raise ValueError(f"Circular 'extends' detected at: {p}")
    seen.add(p)

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        return cfg

    extends = cfg.pop("extends", None)
    if extends is None:
        return cfg

    extends_list = [extends] if isinstance(extends, (str, Path)) else list(extends)
    merged: dict = {}
    for ep in extends_list:
        ep_path = Path(ep)
        if not ep_path.is_absolute():
            ep_path = (p.parent / ep_path).resolve()
        if not ep_path.exists():
            raise FileNotFoundError(f"extends target not found: {ep_path} (from {p})")
        base = load_case_yaml(ep_path, _seen=seen)
        if not isinstance(base, dict):
            raise ValueError(f"extends target {ep_path} must parse to a dict")
        merged = _deep_merge(merged, base)

    return _deep_merge(merged, cfg)


def load_series_csv_1col(path: str | Path, T: int | None = None) -> list[float]:
    path = Path(path)
    out: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p for p in s.replace(";", ",").split(",") if p != ""]
            if len(parts) == 0:
                continue
            try:
                out.append(float(parts[0]))
            except ValueError:
                continue

    if T is None:
        return out

    if len(out) >= T:
        return out[:T]
    if len(out) == 0:
        return [0.0] * T
    return out + [out[-1]] * (T - len(out))


def _pad_or_trunc(arr: Sequence[float], T: int, *, pad_with_last: bool = True) -> List[float]:
    arr = list(arr)
    if len(arr) >= T:
        return arr[:T]
    if len(arr) == 0:
        return [0.0] * T
    pad_val = arr[-1] if pad_with_last else 0.0
    return arr + [pad_val] * (T - len(arr))


def _parse_hhmm(s: str) -> int:
    s = str(s).strip()
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


def _build_time_index(cfg: dict, T: int) -> List[dt.datetime]:
    time_cfg = cfg.get("time", {}) if isinstance(cfg.get("time", {}), dict) else {}
    start = time_cfg.get("start")
    dt_hours = float(time_cfg.get("dt_hours", 1.0))

    if start is None:
        base = dt.datetime(2025, 1, 1, 0, 0)
    elif isinstance(start, dt.datetime):
        base = start
    else:
        s = str(start)
        try:
            base = dt.datetime.fromisoformat(s)
        except ValueError:
            base = dt.datetime.strptime(s, "%Y-%m-%d %H:%M")

    step = dt.timedelta(hours=dt_hours)
    return [base + k * step for k in range(T)]


def _expand_flat(spec: Any, T: int) -> List[float]:
    if spec is None:
        return [0.0] * T
    if isinstance(spec, (int, float)):
        return [float(spec)] * T
    if isinstance(spec, (list, tuple)):
        return _pad_or_trunc([float(x) for x in spec], T)
    if isinstance(spec, dict):
        if "flat" in spec:
            return [float(spec["flat"])] * T
        if "value" in spec:
            return [float(spec["value"])] * T
        return [0.0] * T
    raise TypeError(f"Unsupported tariff spec type: {type(spec)}")


def _expand_bi_horaria(spec: Any, tindex: List[dt.datetime]) -> List[float]:
    T = len(tindex)
    if not isinstance(spec, dict):
        return _expand_flat(spec, T)

    peak = float(spec.get("peak", spec.get("ponta", 0.0)))
    offpeak = float(spec.get("offpeak", spec.get("vazio", 0.0)))

    periods = spec.get("peak_periods")
    if periods is None:
        periods = [["07:00", "10:00"], ["18:00", "22:00"]]

    parsed: List[Tuple[int, int]] = []
    for a, b in periods:
        a_m = _parse_hhmm(a)
        b_m = _parse_hhmm(b)
        parsed.append((a_m, b_m))

    out: List[float] = []
    for ts in tindex:
        m = ts.hour * 60 + ts.minute
        is_peak = False
        for a_m, b_m in parsed:
            if a_m <= b_m:
                if a_m <= m < b_m:
                    is_peak = True
                    break
            else:
                if m >= a_m or m < b_m:
                    is_peak = True
                    break
        out.append(peak if is_peak else offpeak)
    return out


def _expand_monthly(spec: Any, tindex: List[dt.datetime]) -> List[float]:
    T = len(tindex)
    if not isinstance(spec, dict):
        return _expand_flat(spec, T)

    by = spec.get("by_month")
    if not isinstance(by, dict):
        return _expand_flat(spec, T)

    default = float(by.get("default", spec.get("default", 0.0)))
    out: List[float] = []
    for ts in tindex:
        m = ts.month
        v = by.get(str(m), by.get(m, default))
        out.append(float(v))
    return out


def _expand_week_weekend(spec: Any, tindex: List[dt.datetime]) -> List[float]:
    T = len(tindex)
    if not isinstance(spec, dict):
        return _expand_flat(spec, T)

    weekday_spec = spec.get("weekday", 0.0)
    weekend_spec = spec.get("weekend", 0.0)

    weekday_series = _expand_any(weekday_spec, tindex)
    weekend_series = _expand_any(weekend_spec, tindex)

    out: List[float] = []
    for i, ts in enumerate(tindex):
        is_weekend = ts.weekday() >= 5
        out.append(weekend_series[i] if is_weekend else weekday_series[i])
    return out


def _expand_any(spec: Any, tindex: List[dt.datetime], model_hint: Optional[str] = None) -> List[float]:
    T = len(tindex)

    if spec is None or isinstance(spec, (int, float, list, tuple)):
        return _expand_flat(spec, T)

    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported tariff spec type: {type(spec)}")

    if "flat" in spec or "value" in spec:
        return _expand_flat(spec, T)

    if "weekday" in spec and "weekend" in spec:
        return _expand_week_weekend(spec, tindex)

    if "by_month" in spec:
        return _expand_monthly(spec, tindex)

    if ("peak" in spec or "offpeak" in spec or "ponta" in spec or "vazio" in spec) and ("peak_periods" in spec or model_hint in ("bi_horaria", "bihoraria")):
        return _expand_bi_horaria(spec, tindex)

    if model_hint in ("bi_horaria", "bihoraria"):
        return _expand_bi_horaria(spec, tindex)
    if model_hint in ("monthly",):
        return _expand_monthly(spec, tindex)
    if model_hint in ("week_weekend", "weekday_weekend"):
        return _expand_week_weekend(spec, tindex)

    return _expand_flat(spec, T)


def _merged_house_tariff_spec(cfg: dict, hid: str, key: str) -> Any:
    base = (cfg.get("tariffs") or {}).get(key)
    override = (((cfg.get("houses") or {}).get(hid) or {}).get("tariffs") or {}).get(key)
    return override if override is not None else base


def build_tariffs(
    cfg: dict,
    T: int,
    *,
    houses: Optional[Sequence[str]] = None,
    root: Optional[Path] = None,
) -> Tuple[List[float] | Dict[str, List[float]], List[float] | Dict[str, List[float]]]:
    tindex = _build_time_index(cfg, T)
    tariffs_cfg = cfg.get("tariffs", {}) if isinstance(cfg.get("tariffs", {}), dict) else {}
    model_hint = tariffs_cfg.get("model", None)
    if isinstance(model_hint, str):
        model_hint = model_hint.strip().lower()

    def expand(spec: Any) -> List[float]:
        return _expand_any(spec, tindex, model_hint=model_hint)

    if houses is None:
        buy = expand(tariffs_cfg.get("grid_buy", 0.0))
        sell = expand(tariffs_cfg.get("grid_sell", 0.0))
        return buy, sell

    buy_by_house: Dict[str, List[float]] = {}
    sell_by_house: Dict[str, List[float]] = {}
    for hid in houses:
        hid = str(hid)
        spec_buy = _merged_house_tariff_spec(cfg, hid, "grid_buy")
        spec_sell = _merged_house_tariff_spec(cfg, hid, "grid_sell")
        buy_by_house[hid] = expand(spec_buy)
        sell_by_house[hid] = expand(spec_sell)

    return buy_by_house, sell_by_house

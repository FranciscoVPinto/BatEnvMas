from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, time as dtime

import pandas as pd
import yaml


def load_case_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML case file must parse to a dict.")
    cfg["_case_path"] = str(path.resolve())
    return cfg


def load_series_csv_1col(path: str | Path, T: int | None = None) -> list[float]:
    path = Path(path)
    df = pd.read_csv(path, header=None)
    series = df.iloc[:, 0].astype(float).tolist()
    if T is not None:
        if len(series) < T:
            raise ValueError(f"Time series too short: {path} has {len(series)} rows, but T={T}")
        series = series[:T]
    return series


def _as_series_or_scalar(x: Any, T: int) -> list[float]:
    if isinstance(x, (int, float)):
        return [float(x)] * T
    if isinstance(x, list):
        if len(x) < T:
            raise ValueError(f"Provided list length {len(x)} < T={T}")
        return [float(v) for v in x[:T]]
    raise ValueError(f"Unsupported tariff type: {type(x)}")


def _parse_hhmm(s: str) -> dtime:
    try:
        hh, mm = s.strip().split(":")
        return dtime(hour=int(hh), minute=int(mm))
    except Exception as e:
        raise ValueError(f"Invalid time '{s}'. Expected 'HH:MM'.") from e


def _tod_minutes(ts: datetime) -> int:
    return ts.hour * 60 + ts.minute


def _in_period(minute: int, start_min: int, end_min: int) -> bool:
    if start_min == end_min:
        return False
    if start_min < end_min:
        return start_min <= minute < end_min
    return minute >= start_min or minute < end_min


def _compile_bihoraria(spec: dict) -> tuple[float, float, list[tuple[int, int]]]:
    if not isinstance(spec, dict):
        raise ValueError("bi_horaria spec must be a dict with peak/offpeak/peak_periods or flat")
    if "flat" in spec:
        p = float(spec["flat"])
        return p, p, []

    peak = float(spec["peak"])
    offpeak = float(spec["offpeak"])
    periods = spec.get("peak_periods", [])
    if not isinstance(periods, list) or not periods:
        raise ValueError("bi_horaria requires peak_periods: a non-empty list of ['HH:MM','HH:MM'] pairs")

    parsed: list[tuple[int, int]] = []
    for p in periods:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ValueError("Each peak_period must be ['HH:MM','HH:MM']")
        a = _parse_hhmm(str(p[0]))
        b = _parse_hhmm(str(p[1]))
        parsed.append((a.hour * 60 + a.minute, b.hour * 60 + b.minute))
    return peak, offpeak, parsed


def _bihoraria_price_at(peak: float, offpeak: float, periods: list[tuple[int, int]], ts: datetime) -> float:
    if not periods:
        return peak
    m = _tod_minutes(ts)
    is_peak = any(_in_period(m, smin, emin) for smin, emin in periods)
    return peak if is_peak else offpeak


def _build_bihoraria_series(spec: dict, *, T: int, dt_hours: float, start: datetime) -> list[float]:
    peak, offpeak, periods = _compile_bihoraria(spec)
    out: list[float] = []
    step = timedelta(hours=float(dt_hours))
    ts = start
    for _ in range(T):
        out.append(_bihoraria_price_at(peak, offpeak, periods, ts))
        ts += step
    return out


def _build_week_weekend_series(spec: dict, *, T: int, dt_hours: float, start: datetime) -> list[float]:
    if not isinstance(spec, dict):
        raise ValueError("week_weekend spec must be a dict with keys weekday/weekend")

    weekday_spec = spec.get("weekday", None)
    weekend_spec = spec.get("weekend", None)
    if weekday_spec is None or weekend_spec is None:
        raise ValueError("week_weekend requires grid_buy.weekday and grid_buy.weekend specs")

    def _prep(day_spec: Any):
        if isinstance(day_spec, (int, float)):
            return ("flat", float(day_spec))
        if isinstance(day_spec, dict):
            if "flat" in day_spec:
                return ("flat", float(day_spec["flat"]))
            peak, offpeak, periods = _compile_bihoraria(day_spec)
            return ("bih", (peak, offpeak, periods))
        raise ValueError("weekday/weekend spec must be number or dict")

    weekday_compiled = _prep(weekday_spec)
    weekend_compiled = _prep(weekend_spec)

    out: list[float] = []
    step = timedelta(hours=float(dt_hours))
    ts = start
    for _ in range(T):
        is_weekend = ts.weekday() >= 5
        kind, payload = weekend_compiled if is_weekend else weekday_compiled
        if kind == "flat":
            out.append(float(payload))
        else:
            peak, offpeak, periods = payload
            out.append(_bihoraria_price_at(peak, offpeak, periods, ts))
        ts += step
    return out


def _build_monthly_series(spec: dict, *, T: int, dt_hours: float, start: datetime) -> list[float]:
    if not isinstance(spec, dict) or "by_month" not in spec:
        raise ValueError("monthly spec must be a dict with 'by_month' mapping")

    by_month = spec.get("by_month", {})
    if not isinstance(by_month, dict):
        raise ValueError("monthly.by_month must be a dict")

    default = by_month.get("default", None)
    if default is None:
        raise ValueError("monthly.by_month requires a 'default' price")

    def _get_for_month(m: int) -> float:
        if m in by_month:
            return float(by_month[m])
        if str(m) in by_month:
            return float(by_month[str(m)])
        return float(default)

    out: list[float] = []
    step = timedelta(hours=float(dt_hours))
    ts = start
    for _ in range(T):
        out.append(_get_for_month(ts.month))
        ts += step
    return out


def build_tariffs(cfg: dict, T: int) -> tuple[list[float], list[float]]:
    tariffs = cfg.get("tariffs", {})
    if not isinstance(tariffs, dict):
        raise ValueError("tariffs must be a dict")

    time_cfg = cfg.get("time", {}) if isinstance(cfg.get("time", {}), dict) else {}
    dt_hours = float(time_cfg.get("dt_hours", 1.0))

    model = tariffs.get("model", None)
    if not model:
        buy = _as_series_or_scalar(tariffs.get("grid_buy", 0.25), T)
        sell = _as_series_or_scalar(tariffs.get("grid_sell", 0.05), T)
        return buy, sell

    model = str(model).strip().lower()
    start_s = time_cfg.get("start", None)
    if start_s is None:
        if model in ("week_weekend", "monthly"):
            raise ValueError("time.start is required for tariffs.model=week_weekend or monthly (e.g. '2025-01-01 00:00')")
        start = datetime(2000, 1, 1, 0, 0)
    else:
        try:
            start = datetime.fromisoformat(str(start_s))
        except Exception as e:
            raise ValueError("time.start must be ISO format like 'YYYY-MM-DD HH:MM'") from e

    def _series_for(spec: Any, default_scalar: float) -> list[float]:
        if spec is None:
            return [float(default_scalar)] * T
        if isinstance(spec, (int, float, list)):
            return _as_series_or_scalar(spec, T)
        if not isinstance(spec, dict):
            raise ValueError(f"Tariff spec must be number/list/dict, got {type(spec)}")
        if model == "bi_horaria":
            return _build_bihoraria_series(spec, T=T, dt_hours=dt_hours, start=start)
        if model == "week_weekend":
            return _build_week_weekend_series(spec, T=T, dt_hours=dt_hours, start=start)
        if model == "monthly":
            return _build_monthly_series(spec, T=T, dt_hours=dt_hours, start=start)
        raise ValueError(f"Unsupported tariffs.model: {model}")

    buy = _series_for(tariffs.get("grid_buy", None), 0.25)
    sell = _series_for(tariffs.get("grid_sell", None), 0.05)
    return buy, sell

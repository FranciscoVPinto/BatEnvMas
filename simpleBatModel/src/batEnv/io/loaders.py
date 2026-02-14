from __future__ import annotations

from pathlib import Path
from typing import Any

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
    """
    Loads a 1-column CSV with no header.

    Behavior:
      - Reads first column as float.
      - If T is provided:
          * raises ValueError if file length < T (fail fast)
          * otherwise returns the first T samples
    """
    path = Path(path)
    df = pd.read_csv(path, header=None)
    series = df.iloc[:, 0].astype(float).tolist()

    if T is not None:
        if len(series) < T:
            raise ValueError(f"Time series too short: {path} has {len(series)} rows, but T={T}")
        series = series[:T]
    return series


def _as_series_or_scalar(x: Any, T: int) -> list[float]:
    """
    If x is a number -> expand to length T.
    If x is a list -> validate length >= T and truncate.
    """
    if isinstance(x, (int, float)):
        return [float(x)] * T
    if isinstance(x, list):
        if len(x) < T:
            raise ValueError(f"Provided list length {len(x)} < T={T}")
        return [float(v) for v in x[:T]]
    raise ValueError(f"Unsupported tariff type: {type(x)}")


def build_tariffs(cfg: dict, T: int) -> tuple[list[float], list[float]]:
    """
    Supports scalar tariffs:
      tariffs: {grid_buy: 0.25, grid_sell: 0.05}
    or lists of length >= T.
    """
    tariffs = cfg.get("tariffs", {})
    if not isinstance(tariffs, dict):
        raise ValueError("tariffs must be a dict")

    buy = _as_series_or_scalar(tariffs.get("grid_buy", 0.25), T)
    sell = _as_series_or_scalar(tariffs.get("grid_sell", 0.05), T)
    return buy, sell

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
import pandas as pd

# -------- I/O e CSV --------

def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_csvs(outdir: Path, named_frames: Dict[str, pd.DataFrame | pd.Series]) -> None:
    ensure_outdir(outdir)
    for fname, obj in named_frames.items():
        df = obj.to_frame() if isinstance(obj, pd.Series) else obj
        df.to_csv(outdir / fname, index=True)

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def try_parse_timestamps(raw: pd.DataFrame, skip_first_cols: int) -> Optional[pd.DatetimeIndex]:
    if skip_first_cols < 1 or raw.shape[1] == 0:
        return None
    try:
        ts = pd.to_datetime(raw.iloc[:, 0], errors="raise", utc=False, infer_datetime_format=True)
        return pd.DatetimeIndex(ts)
    except Exception:
        return None

def load_wide_csv_with_ts(path: Path, skip_first_cols: int) -> Tuple[pd.DataFrame, Optional[pd.Index]]:
    raw = pd.read_csv(path)
    ts = try_parse_timestamps(raw, skip_first_cols)
    data = raw.iloc[:, skip_first_cols:].copy()
    data = coerce_numeric(data)
    if ts is not None and len(ts) == len(data):
        data.index = ts
    return data, (data.index if ts is not None else None)

def take_first_cols(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    if n is None or n >= df.shape[1]:
        return df
    return df.iloc[:, :max(0, int(n))].copy()

# -------- Janela temporal --------

def slice_week(df: pd.DataFrame, interval_min: int, week: int, week_offset: int, horizon: int) -> pd.DataFrame:
    rows_per_day = int(24 * 60 // interval_min)
    rows_per_week = rows_per_day * 7
    w_idx = max(1, int(week) + int(week_offset))
    start = (w_idx - 1) * rows_per_week
    stop = start + rows_per_week
    if start >= len(df):
        stop = len(df)
        start = max(0, stop - rows_per_week)
    dfw = df.iloc[start:stop].copy()
    if horizon and 0 < horizon < len(dfw):
        dfw = dfw.iloc[:horizon].copy()
    return dfw

# -------- Remapeamentos A1/A2/... → nomes reais --------

def normalize_agent_key(key, agents: List[str]) -> Optional[str]:
    k = str(key)
    if k in agents:
        return k
    m = re.match(r"^[Aa](\d+)$", k)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(agents):
            return agents[idx]
    return None

def remap_shares(shares: Dict[str, float], agents: List[str]) -> Dict[str, float]:
    remapped: Dict[str, float] = {}
    missing: List[str] = []
    for k, v in shares.items():
        nk = normalize_agent_key(k, agents)
        if nk is None:
            missing.append(k)
        else:
            remapped[nk] = float(v)
    if missing:
        print(f"[WARN] Ignoring unknown agent keys in shares: {missing}. Available: {agents}")
    return remapped

def remap_groups(groups: Dict[str, List[str]], agents: List[str]) -> Dict[str, List[str]]:
    new_groups: Dict[str, List[str]] = {}
    for g, members in groups.items():
        resolved: List[str] = []
        missing: List[str] = []
        for m in members:
            nm = normalize_agent_key(m, agents)
            if nm is None:
                missing.append(m)
            else:
                resolved.append(nm)
        if missing:
            print(f"[WARN] In group '{g}', ignoring unknown members: {missing}. Available: {agents}")
        if resolved:
            new_groups[g] = resolved
    new_groups = {g: ms for g, ms in new_groups.items() if ms}
    if not new_groups:
        raise ValueError("All hierarchical groups became empty after remapping. Check names.")
    return new_groups

def remap_windows(
    windows: List[tuple[int | str, int | str, Dict[str, float]]],
    agents: List[str],
    H: int
) -> List[tuple[int, int, Dict[str, float]]]:
    def _resolve_edge(edge) -> int:
        if edge == "START": return 0
        if edge == "MID":   return H // 2
        if edge == "END":   return H
        return max(0, min(H, int(edge)))
    out: List[tuple[int, int, Dict[str, float]]] = []
    for (t0, t1, sh) in windows:
        a = _resolve_edge(t0)
        b = _resolve_edge(t1)
        if b < a:
            a, b = b, a
        a = max(0, min(a, H))
        b = max(0, min(b, H))
        if a == b:
            continue
        out.append((a, b, remap_shares(sh, agents)))
    if not out:
        raise ValueError("All dynamic windows collapsed to empty after remapping.")
    return out

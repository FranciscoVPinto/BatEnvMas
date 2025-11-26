from __future__ import annotations

import numpy as np
import pandas as pd

def eval_metrics(pv_global: pd.Series, load_global: pd.Series,
                 pv_used: pd.Series, export_series: pd.Series, import_series: pd.Series) -> dict:
    eps = 1e-12
    sc = pv_used.sum() / (pv_global.sum() + eps)          # Self-consumption (PV used / PV gen)
    ss = pv_used.sum() / (load_global.sum() + eps)        # Self-sufficiency (PV used / Load)
    exp_ratio = export_series.sum() / (pv_global.sum() + eps)
    imp_ratio = import_series.sum() / (load_global.sum() + eps)
    return {"SelfCons": sc, "SelfSuff": ss, "Export/PV": exp_ratio, "Import/Load": imp_ratio}


def metrics_table(summary: dict) -> pd.DataFrame:
    """summary[name] -> dict de métricas -> DataFrame ordenado por SelfCons desc."""
    df = pd.DataFrame(summary).T
    df = df[["SelfCons","SelfSuff","Export/PV","Import/Load"]]
    return df.sort_values("SelfCons", ascending=False)


def fairness_index(pv_alloc_df: pd.DataFrame, load_df: pd.DataFrame) -> float:
    """Returns fairness index based on allocation vs load (per-agent ratio)."""
    eps = 1e-12  # to avoid division by zero
    total_alloc = pv_alloc_df.sum()
    total_load = load_df.sum()
    ratios = total_alloc / (total_load + eps)
    mean = ratios.mean()
    std = ratios.std()
    val = 1 - (std / (mean + eps))  # 1 = perfeita; pode ficar <0 se dispersão >> média
    return max(0.0, float(val))


def jains_index(pv_alloc_df: pd.DataFrame, load_df: pd.DataFrame) -> float:
    ratios = (pv_alloc_df.sum() / (load_df.sum() + 1e-12)).values
    numerator = np.sum(ratios)**2
    denominator = len(ratios) * np.sum(ratios**2) + 1e-12
    return numerator / denominator  # ∈ (0,1]


def community_energy_flows(pv_global: pd.Series, load_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    load_global = load_df.sum(axis=1)
    pv_used = pd.Series(np.minimum(pv_global.values, load_global.values), index=pv_global.index)
    export_series = pd.Series(np.maximum(pv_global.values - load_global.values, 0.0), index=pv_global.index)
    import_series = pd.Series(np.maximum(load_global.values - pv_global.values, 0.0), index=pv_global.index)
    return pv_used, export_series, import_series


def energy_kpis(load_df: pd.DataFrame, alloc_df: pd.DataFrame, pv_series: pd.Series) -> pd.DataFrame:
    load_global = load_df.sum(axis=1)
    pv_used, export_series, import_series = community_energy_flows(pv_series, load_df)
    m = eval_metrics(pv_global=pv_series, load_global=load_global,
                     pv_used=pv_used, export_series=export_series, import_series=import_series)
    return pd.DataFrame([m])

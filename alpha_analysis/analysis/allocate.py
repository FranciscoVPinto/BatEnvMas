from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

from analysis.alphas import normalize_alpha_cols  # column-stochastic safeguard


def allocate_pv_global_to_load(
    pv_global: pd.Series,          # (H,)
    load_df: pd.DataFrame,         # (H x n)
    alpha: np.ndarray,             # (n x H) coluna-estocástico
    agents: List[str],
    tol: float = 1e-9,
    max_rounds: int = 5,           # ignorado nesta variante (sem redistribuição)
    eligible_mask: np.ndarray | None = None,  # (n x H) True = pode receber
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    n = len(agents)
    H = len(pv_global)
    if alpha.shape != (n, H):
        raise ValueError(f"alpha shape {alpha.shape} != ({n}, {H})")
    if load_df.shape != (H, n):
        raise ValueError(f"load_df shape {load_df.shape} != ({H}, {n})")
    if eligible_mask is not None and eligible_mask.shape != (n, H):
        raise ValueError(f"eligible_mask shape {eligible_mask.shape} != ({n}, {H})")

    alloc = np.zeros((H, n), dtype=float)
    pv_vals = pv_global.values

    for t in range(H):
        pv_t = pv_vals[t]
        if pv_t <= tol:
            continue

        # 1) pesos base
        a = alpha[:, t].astype(float).copy()

        # 2) aplicar elegibilidade (se existir)
        if eligible_mask is not None:
            a[~eligible_mask[:, t]] = 0.0

        # 3) normalizar por coluna (se soma > 0); caso contrário, ninguém recebe
        s = a.sum()
        if s > tol:
            w = a / s
        else:
            # ninguém elegível / pesos nulos -> sem alocação neste t
            continue

        # 4) Atribuição pura (SEM cap pelo load)
        alloc[t, :] = pv_t * w

    # DataFrame da alocação
    alloc_df = pd.DataFrame(alloc, columns=agents, index=load_df.index)

    consumed_clamped = np.minimum(load_df.values, alloc_df.values)
    consumed_clamped_sum = consumed_clamped.sum(axis=1)

    export_series = pd.Series(
        (pv_vals - consumed_clamped_sum).clip(min=0.0),
        index=load_df.index,
        name="export_physical",
    )

    total_load_t = load_df.values.sum(axis=1)
    import_series = pd.Series(
        (total_load_t - consumed_clamped_sum).clip(min=0.0),
        index=load_df.index,
        name="import_physical",
    )

    return alloc_df, export_series, import_series


def allocate_unlimited(pv_global: pd.Series, alpha: np.ndarray, agents: List[str]) -> pd.DataFrame:
    """Distribute 100% of PV_global(t) by alpha[:, t] with no caps or eligibility.
    Matches the runner's expected signature and returns only the allocation DataFrame.
    """
    H = len(pv_global)
    n = len(agents)
    if alpha.shape != (n, H):
        raise ValueError(f"alpha shape {alpha.shape} != ({n}, {H})")
    alpha = normalize_alpha_cols(alpha)
    alloc = (alpha * pv_global.to_numpy()[None, :]).T  # (H, n)
    return pd.DataFrame(alloc, columns=agents, index=pv_global.index)


def build_receiver_mask(load_df: pd.DataFrame, pv_df: pd.DataFrame, tol: float = 1e-9) -> np.ndarray:
    """Elegibilidade para receber partilha no período:
    True = pode receber; False = está a injetar (gen - load > 0).
    Returns shape (n x H) to match alpha's orientation.
    """
    net = pv_df.values - load_df.values   # (H x n)
    inject = (net > tol)                  # (H x n)
    eligible = ~inject
    return eligible.T.astype(bool)        # (n x H)

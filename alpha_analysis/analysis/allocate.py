from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

def allocate_pv_global_to_load(
    pv_global: pd.Series,          # (H,)
    load_df: pd.DataFrame,         # (H x n)
    alpha: np.ndarray,             # (n x H) coluna-estocástico
    agents: List[str],
    tol: float = 1e-9,
    max_rounds: int = 5,
    eligible_mask: np.ndarray | None = None,  # (n x H) True = pode receber
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Aloca PV_global(t) às cargas L_i(t) com cap x_i(t) ≤ L_i(t),
    com redistribuição iterativa e elegibilidade opcional.

    Returns:
      - alloc_df: (H x n) PV usado por agente no instante (kWh por slot)
      - export_series: (H,) PV que sobrou (excedente exportado)
      - import_series: (H,) carga que faltou cobrir (importação da rede)
    """
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
    load_vals = load_df.values

    for t in range(H):
        avail = pv_vals[t]
        if avail <= tol:
            continue

        a = alpha[:, t].copy()
        remaining = load_vals[t].copy()

        if eligible_mask is not None:
            # zera quem não pode receber e cap nas necessidades
            a[~eligible_mask[:, t]] = 0.0
            remaining[~eligible_mask[:, t]] = 0.0

        s = a.sum()
        a[:] = (a / s) if s > 0 else (1.0 / n)

        active = remaining > tol
        rounds = 0
        while avail > tol and active.any() and rounds < max_rounds:
            weight_sum = a[active].sum()
            if weight_sum <= 0:
                weights = np.zeros_like(a)
                weights[active] = 1.0 / active.sum()
            else:
                weights = np.zeros_like(a)
                weights[active] = a[active] / weight_sum

            proposed = avail * weights
            give = np.minimum(proposed, remaining)
            got = give.sum()

            alloc[t, :] += give
            remaining -= give
            avail -= got
            active = remaining > tol
            rounds += 1

    alloc_df = pd.DataFrame(alloc, columns=agents, index=load_df.index)
    pv_used = alloc_df.sum(axis=1)
    export_series = (pv_global - pv_used).clip(lower=0.0)
    import_series = (load_df.sum(axis=1) - pv_used).clip(lower=0.0)
    return alloc_df, export_series, import_series

def build_receiver_mask(load_df: pd.DataFrame, pv_df: pd.DataFrame, tol: float=1e-9) -> np.ndarray:
    """
    Elegibilidade para receber partilha no período:
    True = pode receber; False = está a injetar (gen - load > 0).
    """
    net = pv_df.values - load_df.values   # (H x n)
    inject = (net > tol)                  # (H x n)
    eligible = ~inject
    return eligible.T.astype(bool)        # (n x H)

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

# ========= Base builders =========
def alpha_equal(n_agents: int, H: int) -> np.ndarray:
    """Divide sempre por igual: α_i(t) = 1/n."""
    return np.full((n_agents, H), 1.0 / n_agents, dtype=float)

def alpha_proportional_mean_load(load_df: pd.DataFrame) -> np.ndarray:
    """Constante no tempo: pesos ∝ carga média de cada agente."""
    H = load_df.shape[0]
    means = load_df.mean(axis=0).values  # (n,)
    total = means.sum()
    if total <= 0:
        return alpha_equal(load_df.shape[1], H)
    w = means / total
    return np.tile(w.reshape(-1, 1), (1, H))  # (n x H)

def alpha_instant_load_share(load_df: pd.DataFrame) -> np.ndarray:
    """Proporcional à carga instantânea: α_i(t) ∝ L_i(t)."""
    L = load_df.values                   # (H x n)
    denom = L.sum(axis=1, keepdims=True) # (H x 1)
    share = np.divide(L, denom, out=np.zeros_like(L), where=denom > 0)  # (H x n)
    return share.T  # (n x H)

def normalize_alpha_cols(alpha: np.ndarray) -> np.ndarray:
    """Garante colunas a somar 1; colunas nulas viram split igual."""
    n, H = alpha.shape
    out = alpha.copy()
    col_sums = out.sum(axis=0, keepdims=True)   # (1, H)
    nz_cols = (col_sums > 0).ravel()            # (H,)
    out[:, nz_cols] = out[:, nz_cols] / col_sums[0, nz_cols]
    if (~nz_cols).any():
        out[:, ~nz_cols] = 1.0 / n
    return out

# ========= Fixed (RAC Art. 29.º) =========
def alpha_fixed_from_shares(shares: Dict[str, float], agents: List[str], H: int) -> np.ndarray:
    """Coeficientes fixos constantes no tempo (somam 1)."""
    w = np.array([max(0.0, shares.get(a, 0.0)) for a in agents], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(agents), dtype=float)
    w /= w.sum()
    return np.tile(w.reshape(-1, 1), (1, H))

def alpha_fixed_from_df(alpha_df: pd.DataFrame, agents: List[str], H: int) -> np.ndarray:
    """Time-varying: alpha_df (H x n) com colunas = agents. Normaliza por coluna depois."""
    if list(alpha_df.columns) != agents:
        alpha_df = alpha_df.reindex(columns=agents)
    if len(alpha_df) != H:
        raise ValueError(f"alpha_fixed_from_df: H mismatch ({len(alpha_df)} vs {H})")
    return alpha_df.T.clip(lower=0.0).values

# ========= Hierarchical (RAC Art. 31.º) =========
def alpha_hierarchical(
    groups: Dict[str, List[str]],
    agents: List[str],
    H: int,
    inner_rule: str = "InstantLoad",       # "InstantLoad" | "Equal" | "ProportionalMean"
    outer_shares: Dict[str, float] | None = None,
    load_df: pd.DataFrame | None = None
) -> np.ndarray:
    """
    alpha_agent(t) = alpha_within_group(t) * alpha_group
    - dentro do grupo: inner_rule
    - entre grupos: outer_shares (fixo; normalizado)
    """
    n = len(agents)
    A = np.zeros((n, H), dtype=float)

    # Entre grupos (fixo)
    if outer_shares is None:
        outer_shares = {g: 1.0 for g in groups}
    s = sum(max(0.0, v) for v in outer_shares.values())
    outer = {g: (max(0.0, outer_shares.get(g, 0.0)) / s if s > 0 else 1.0/len(groups))
             for g in groups}

    # Dentro de cada grupo
    for g, members in groups.items():
        idx = [agents.index(a) for a in members]

        if inner_rule == "Equal":
            inner = np.full((len(members), H), 1.0/len(members), dtype=float)
        elif inner_rule == "ProportionalMean":
            assert load_df is not None, "load_df necessário para ProportionalMean"
            inner = alpha_proportional_mean_load(load_df[members])
        elif inner_rule == "InstantLoad":
            assert load_df is not None, "load_df necessário para InstantLoad"
            inner = alpha_instant_load_share(load_df[members])
        else:
            raise ValueError(f"inner_rule desconhecida: {inner_rule}")

        inner = normalize_alpha_cols(inner)
        A[idx, :] += inner * outer[g]

    return normalize_alpha_cols(A)

# ========= Dynamic (RAC Art. 32.º, ex-post by windows) =========
def alpha_dynamic_windows(
    agents: List[str],
    H: int,
    windows: list[tuple[int, int, Dict[str, float]]]
) -> np.ndarray:
    """
    windows: [(t0, t1_excl, shares_dict_benef)]
    Ex.: [(0,96*7,{"A01":0.6,"A02":0.4}), (96*7,H,{"A01":0.3,"A02":0.7})]
    """
    A = np.zeros((len(agents), H), dtype=float)
    for (t0, t1, shares) in windows:
        vec = np.zeros(len(agents), dtype=float)
        for i, a in enumerate(agents):
            vec[i] = max(0.0, shares.get(a, 0.0))
        if vec.sum() <= 0:
            vec[:] = 1.0/len(agents)
        else:
            vec /= vec.sum()
        A[:, t0:t1] = vec.reshape(-1, 1)
    return normalize_alpha_cols(A)

# ========= Convenience: build standard set =========
def build_alpha_configs_pv_global(load_df: pd.DataFrame) -> dict:
    H = load_df.shape[0]
    n = load_df.shape[1]
    cfgs = {
        "Equal":        alpha_equal(n, H),
        "Proportional": alpha_proportional_mean_load(load_df),
        "InstantLoad":  alpha_instant_load_share(load_df),
    }
    return {k: normalize_alpha_cols(v) for k, v in cfgs.items()}

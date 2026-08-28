"""
Re-amostragem das séries de entrada.

Todos os CSV do projecto estão gravados a 15 min (`NATIVE_DT_HOURS`). Quando um
caso pede um passo temporal mais grosseiro, as séries são reduzidas por média de
blocos, o que preserva a energia total (kWh) quando combinada com o novo `dt`.

Movido de `scripts/run_case.py` sem alterações de comportamento.
"""
from __future__ import annotations

from typing import Sequence

# Resolução nativa de TODOS os CSV de entrada (15 min).
NATIVE_DT_HOURS = 0.25


def resample_series(series: Sequence[float], factor: int) -> list[float]:
    """Reduz a série por média de `factor` valores consecutivos.

    Adequado a séries de potência (kW) ou de preço (EUR/kWh): a média preserva
    o total em kWh quando combinada com o novo (maior) `dt`.

    Exemplos
    --------
    factor=4 : 15 min -> 1 h    (dt passa de 0.25 para 1.0)
    factor=2 : 15 min -> 30 min
    """
    if factor <= 1:
        return list(series)
    n_full = (len(series) // factor) * factor
    return [sum(series[i: i + factor]) / factor for i in range(0, n_full, factor)]


def resample_factor(dt_hours: float) -> int:
    """Factor de redução para passar de `NATIVE_DT_HOURS` a `dt_hours`."""
    return max(1, round(float(dt_hours) / NATIVE_DT_HOURS))

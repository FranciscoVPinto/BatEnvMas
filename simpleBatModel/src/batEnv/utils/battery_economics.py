"""
Custo de degradacao da bateria a partir de parametros fisicos.

Formula base (curva de Wohler):
    N_actual = N_rated * (DoD_rated / DoD_actual) ^ gamma
    lambda   = battery_cost / (N_actual * E_usable * (1/eta_ch + eta_dis))
"""

from __future__ import annotations


# Escalaes oficiais de Potencia Contratada em Portugal (ERSE), em kVA.
PORTUGUESE_CONTRACTED_POWER_KVA = [
    1.15, 2.30, 3.45, 4.60, 5.75, 6.90,
    10.35, 13.80, 17.25, 20.70, 27.60, 34.50, 41.40,
]


def nearest_contracted_power(power_kw: float) -> float:
    """Devolve o escalao de PC imediatamente >= power_kw (kVA)."""
    for pc in PORTUGUESE_CONTRACTED_POWER_KVA:
        if pc >= power_kw - 1e-6:
            return pc
    return PORTUGUESE_CONTRACTED_POWER_KVA[-1]


def compute_degradation_cost_per_kwh(
    *,
    battery_cost_eur: float,
    E_max_kwh: float,
    E_min_kwh: float,
    N_rated_cycles: float,
    DoD_rated: float = 0.80,
    aging_exponent: float = 1.50,
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
) -> float:
    """
    Calcula o custo de degradacao lambda (EUR/kWh de throughput AC).

    Usar directamente como `lambda_deg` no modelo de degradacao.

    Parametros
    ----------
    battery_cost_eur  Custo de substituicao da bateria (EUR).
    E_max_kwh         Capacidade maxima (kWh).
    E_min_kwh         Energia minima mantida (kWh). DoD_actual = (E_max-E_min)/E_max.
    N_rated_cycles    Ciclos nominais a profundidade DoD_rated (end-of-life a 80%).
    DoD_rated         DoD a qual N_rated e especificado (default 0.80).
    aging_exponent    Expoente gamma da curva de Wohler (default 1.50 para Li-Ion NMC).
    eta_ch / eta_dis  Eficiencias de carga e descarga (default 0.95).
    """
    if battery_cost_eur <= 0:
        raise ValueError(f"battery_cost_eur deve ser > 0, obtido {battery_cost_eur}")
    if not (0 < E_min_kwh < E_max_kwh):
        raise ValueError(f"Deve verificar-se 0 < E_min ({E_min_kwh}) < E_max ({E_max_kwh})")
    if N_rated_cycles <= 0:
        raise ValueError(f"N_rated_cycles deve ser > 0, obtido {N_rated_cycles}")
    if not (0 < DoD_rated <= 1.0):
        raise ValueError(f"DoD_rated deve estar em (0, 1], obtido {DoD_rated}")

    E_usable   = E_max_kwh - E_min_kwh
    DoD_actual = E_usable / E_max_kwh

    # Vida em ciclos ajustada ao DoD real (curva de Wohler)
    N_actual = N_rated_cycles * (DoD_rated / DoD_actual) ** aging_exponent

    # Throughput AC por ciclo: energia a entrada (carga) + energia a saida (descarga)
    throughput_per_cycle = E_usable * (1.0 / eta_ch + eta_dis)

    return battery_cost_eur / (N_actual * throughput_per_cycle)


def compute_pwl_lambda_by_bin(
    *,
    battery_cost_eur: float,
    E_max_kwh: float,
    N_rated_cycles: float,
    soc_breakpoints: list[float] | None = None,
    DoD_rated: float = 0.95,
    aging_exponent: float = 1.50,
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
) -> list[float]:
    """
    Deriva o lambda por bin de SoC a partir da curva de Wohler (em vez de valores
    manuais). Mantem a coerencia com `compute_degradation_cost_per_kwh`: avaliado
    no DoD efectivo da bateria, o resultado iguala o lambda global.

    Modelo
    ------
    Para um ciclo de profundidade DoD (fraccao de E_max):
        N(DoD)   = N_rated * (DoD_rated/DoD)^gamma
        tpc(DoD) = (DoD * E_max) * (1/eta_ch + eta_dis)        [throughput AC/ciclo]
        lambda(DoD) = cost / (N(DoD) * tpc(DoD)) = C * DoD^(gamma-1)
        C = cost / (N_rated * DoD_rated^gamma * E_max * (1/eta_ch + eta_dis))

    A cada bin de SoC [a, b] associa-se um DoD representativo igual a profundidade
    desde a plena carga ate ao ponto medio do bin: DoD_k = 1 - (a+b)/2. Bins de SoC
    baixo (descarga profunda) ficam mais caros; bins de SoC alto mais baratos. Nota:
    uma curva de Wohler pura (so DoD) NAO inclui penalizacao de SoC elevado.
    """
    if soc_breakpoints is None:
        soc_breakpoints = [0.0, 0.2, 0.8, 1.0]
    if battery_cost_eur <= 0 or E_max_kwh <= 0 or N_rated_cycles <= 0:
        raise ValueError("battery_cost_eur, E_max_kwh e N_rated_cycles devem ser > 0")

    fac = 1.0 / eta_ch + eta_dis
    C = battery_cost_eur / (N_rated_cycles * (DoD_rated ** aging_exponent) * E_max_kwh * fac)

    lambdas: list[float] = []
    for k in range(len(soc_breakpoints) - 1):
        mid = 0.5 * (soc_breakpoints[k] + soc_breakpoints[k + 1])
        dod = max(1.0 - mid, 1e-6)
        lambdas.append(round(C * dod ** (aging_exponent - 1.0), 4))
    return lambdas


def degradation_summary(
    *,
    battery_cost_eur: float,
    E_max_kwh: float,
    E_min_kwh: float,
    N_rated_cycles: float,
    DoD_rated: float = 0.80,
    aging_exponent: float = 1.50,
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
) -> dict:
    """Devolve resumo dos parametros de degradacao, util para registar em meta.yaml."""
    E_usable   = E_max_kwh - E_min_kwh
    DoD_actual = E_usable / E_max_kwh
    N_actual   = N_rated_cycles * (DoD_rated / DoD_actual) ** aging_exponent
    throughput = E_usable * (1.0 / eta_ch + eta_dis)
    lambda_deg = compute_degradation_cost_per_kwh(
        battery_cost_eur=battery_cost_eur, E_max_kwh=E_max_kwh, E_min_kwh=E_min_kwh,
        N_rated_cycles=N_rated_cycles, DoD_rated=DoD_rated,
        aging_exponent=aging_exponent, eta_ch=eta_ch, eta_dis=eta_dis,
    )
    return {
        "battery_cost_eur":         battery_cost_eur,
        "E_usable_kwh":             round(E_usable, 4),
        "DoD_actual":               round(DoD_actual, 4),
        "DoD_rated":                DoD_rated,
        "N_rated_cycles":           N_rated_cycles,
        "N_actual_cycles":          round(N_actual, 1),
        "aging_exponent_gamma":     aging_exponent,
        "throughput_per_cycle_kwh": round(throughput, 4),
        "lifetime_throughput_kwh":  round(N_actual * throughput, 1),
        "lambda_deg_eur_per_kwh":   round(lambda_deg, 6),
    }

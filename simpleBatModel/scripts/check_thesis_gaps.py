"""
Fecha as duas lacunas metodologicas da tese, a partir dos resultados ja gravados.

1) VALOR DA PARTILHA, CONTROLADO PELO ARMAZENAMENTO
   pv21 (PV proprio + baterias reais) vs pv09 (PV partilhado + baterias reais).
   Ambos com 27.45 kWh instalados => a diferenca e atribuivel apenas a partilha.
   (O pv11 usa uma bateria generica de 10 kWh por casa, ~80 kWh no edificio, por
   isso a comparacao pv11-vs-pv09 confunde partilha com dimensionamento.)

2) VALIDACAO EXATA-VS-ROLLING
   Compara results_pwl_metrics.csv e o custo de rede entre o solve exato a 3
   meses (results/full_horizon_sweep/3months/) e o solve com rolling horizon
   (results/pwl_validation/3months_rolling/), para pv15 (alocacao fixa) e
   pv19 (alocacao optimal — a stage 1 que faltava validar).

Correr depois de `run_experiment.py`:
    %runfile scripts/check_thesis_gaps.py --wdir
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DT = 0.25
APTS = [f"Apt{i}" for i in range(1, 9)]


def _house_costs(case_dir: Path) -> dict[str, float] | None:
    """Custo liquido de rede por casa (EUR) = sum((c_grid*P_imp - c_sell*P_exp)*dt)."""
    if not case_dir.is_dir():
        return None
    out: dict[str, float] = {}
    for h in APTS:
        f = case_dir / f"results_house_{h}.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f, usecols=["P_imp", "P_exp", "c_grid", "c_sell"])
        out[h] = float(((df.c_grid * df.P_imp - df.c_sell * df.P_exp) * DT).sum())
    return out


def _fmt(v: float) -> str:
    return f"{v:8.1f}"


def sharing_value_controlled(horizon: str = "1year") -> None:
    base = ROOT / "results" / "full_horizon_sweep" / horizon
    cases = {
        "pv21 own PV + REAL batteries (controlled)": f"b8_pv21_no_sharing_real_batteries_export__{horizon}",
        "pv11 own PV + generic 10 kWh battery": f"b8_pv11_no_sharing_export__{horizon}",
        "pv09 shared PV (optimal) + REAL batteries": f"b8_pv09_optimal_export__{horizon}",
        "pv03 shared PV (instant) + REAL batteries": f"b8_pv03_consumption_instant_export__{horizon}",
        "pv12 no battery, no sharing (baseline)": f"b8_pv12_nothing_export__{horizon}",
    }
    print("=" * 78)
    print(f"1) VALOR DA PARTILHA, CONTROLADO PELO ARMAZENAMENTO  ({horizon})")
    print("=" * 78)
    tot: dict[str, float] = {}
    for label, d in cases.items():
        c = _house_costs(base / d)
        if c is None:
            print(f"  [em falta] {label:44s} -> {d}")
            continue
        tot[label] = sum(c.values())
        print(f"  {label:44s} {_fmt(tot[label])} EUR")

    k21 = "pv21 own PV + REAL batteries (controlled)"
    k09 = "pv09 shared PV (optimal) + REAL batteries"
    k11 = "pv11 own PV + generic 10 kWh battery"
    print()
    if k21 in tot and k09 in tot:
        d = tot[k21] - tot[k09]
        print(f"  >> Efeito da PARTILHA, armazenamento constante (pv21 -> pv09): "
              f"{d:.1f} EUR/ano  ({100*d/tot[k21]:.1f}%)")
    if k11 in tot and k09 in tot:
        d = tot[k11] - tot[k09]
        print(f"  (para comparacao, a versao NAO controlada pv11 -> pv09):      "
              f"{d:.1f} EUR/ano  ({100*d/tot[k11]:.1f}%)")
    if k21 in tot and k11 in tot:
        print(f"  (efeito de sobredimensionar a bateria sem partilha, pv21 -> pv11): "
              f"{tot[k21]-tot[k11]:.1f} EUR/ano)")
    print()


def rolling_validation() -> None:
    exact_root = ROOT / "results" / "full_horizon_sweep" / "3months"
    roll_root = ROOT / "results" / "pwl_validation" / "3months_rolling"
    pairs = [
        ("pv15 equal (fixed alpha)", "b8_pv15_equal_pwl_export__3months",
         "b8_pv15_equal_pwl_export__3months_rolling"),
        ("pv19 optimal (optimal alpha)", "b8_pv19_optimal_pwl_export__3months",
         "b8_pv19_optimal_pwl_export__3months_rolling"),
    ]
    print("=" * 78)
    print("2) VALIDACAO EXATA-VS-ROLLING (3 meses)")
    print("=" * 78)
    print(f"  {'cenario':30s}{'metrica':26s}{'exato':>11s}{'rolling':>11s}{'desvio':>10s}")
    for label, ex, ro in pairs:
        ed, rd = exact_root / ex, roll_root / ro
        if not ed.is_dir() or not rd.is_dir():
            print(f"  [em falta] {label:28s} exato={ed.is_dir()} rolling={rd.is_dir()}")
            continue
        for metric, col in (("custo degradacao (EUR)", "pwl_degradation_cost_EUR"),
                            ("throughput (kWh)", "battery_throughput_kWh")):
            fe, fr = ed / "results_pwl_metrics.csv", rd / "results_pwl_metrics.csv"
            if not (fe.exists() and fr.exists()):
                continue
            e = pd.read_csv(fe)[col].sum()
            r = pd.read_csv(fr)[col].sum()
            dev = 100 * (r - e) / e if e else float("nan")
            print(f"  {label:30s}{metric:26s}{e:11.3f}{r:11.3f}{dev:9.3f}%")
        ce, cr = _house_costs(ed), _house_costs(rd)
        if ce and cr:
            e, r = sum(ce.values()), sum(cr.values())
            print(f"  {label:30s}{'custo de rede (EUR)':26s}{e:11.3f}{r:11.3f}"
                  f"{100*(r-e)/e:9.3f}%")
    print()
    print("  Desvio pequeno (<< 0.5%, o MIP gap usado) => rolling horizon credivel.")
    print()


def main() -> int:
    horizon = sys.argv[1] if len(sys.argv) > 1 else "1year"
    sharing_value_controlled(horizon)
    rolling_validation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

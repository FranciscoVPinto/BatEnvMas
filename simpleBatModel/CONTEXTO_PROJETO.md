# Contexto do projeto — simpleBatModel (handoff)

> Documento de contexto para retomar o trabalho numa conversa nova sem ter de reler
> todo o código. Última atualização: 2026-07-25.

## 1. O que é

Modelo de otimização (MILP, Pyomo + HiGHS) para uma **comunidade de energia** de
8 apartamentos (Edifício 8 / `building8`) com **PV partilhado** e **bateria
doméstica por apartamento**. Objetivo: minimizar o custo líquido de energia da
comunidade, comparando estratégias de partilha de PV e o efeito da degradação da
bateria. Contexto: tese de Engenharia Mecânica.

- Pasta raiz: `C:\Users\pinto\OneDrive - ...\Tese\BatEnvMas\simpleBatModel`
- Ambiente: Anaconda env `lbf`, Python 3.11, Pyomo, solver `appsi_highs` (highspy). Corre no Spyder via `%runfile scripts/run_experiment.py --wdir`.
- Dados: séries a 15 min (`dt_hours=0.25`); 1 ano = 35040 timesteps. `data/load_cons_01..08.csv`, `data/pv_gen_01..08.csv`.

## 2. Estrutura

```
src/batEnv/
  models/    battery.py, multi_house.py (base), multi_house_degradation.py (linear),
             multi_house_degradation_pwl.py (PWL SoC-dependente)
  io/        case_schema.py, orchestration_schemas.py, validate.py, loaders.py, pv_sharing.py
  utils/     battery_economics.py (Wöhler λ), rolling_horizon.py (NOVO), export.py,
             community_metrics.py, solve.py
  plotting/  simple.py, compare.py
scripts/     run_case.py (~920 linhas, motor), run_experiment.py (orquestração de sweeps),
             render_results.py, summarize.py, _common.py (deep_merge, etc.)
cases/       runset_parent.yaml, plotset_parent.yaml
  runsets/   runset_alfa_deg_sweep.yaml, runset_full_horizon_sweep.yaml,
             runset_pwl_sweep.yaml, runset_pwl_validation.yaml (NOVO)
  plotsets/  plotset_alfa_deg_sweep.yaml, plotset_full_horizon_sweep.yaml
  scenarios/building8/  _b8_base.yaml + b8_pv01..b8_pv20 (.yaml)
data/        load_cons_*, pv_gen_*, battery_values_load_01_to_08.csv, max_values_*
results/     alfa_deg_sweep/, full_horizon_sweep/, pwl_validation/
```

## 3. Modelos (formulação)

**Base — `MultiHouseModel`** (`multi_house.py`). Um único binário `y[h,t]`:
- `y=0` modo geração: bateria carrega de excedente PV, pode exportar; `P_imp=0`, `P_dis=0`.
- `y=1` modo consumo: bateria descarrega, pode importar; `P_exp=0`, `P_ch=0`.
- **A bateria nunca carrega da rede** (imposto pelo `y`; por isso `y` NÃO pode ser relaxado a contínuo — na tarifa bi-horária abriria arbitragem de rede).
- Balanço nodal por casa; `E[t]=E[t-1]+η_ch·P_ch·dt − P_dis/η_dis·dt`; limites SoC; `cyclic_soc` impõe `E[T]≥E_init`; `P_imp,P_exp ≤ P_contracted`; `allow_export`.
- `alpha_mode`: `fixed` (PV por casa dado) ou `optimal` (o MILP aloca PV; `Σ_h PV[h,t]=PV_total[t]`).

**Degradação linear — `MultiHouseModelDegradation`**: objetivo + `λ·Σ(P_ch+P_dis)·dt`.

**Degradação PWL — `MultiHouseModelDegradationPWL`**: custo dependente do SoC, K bins,
binários `z[h,t,k]` + McCormick. O modelo monolítico **não escala** (~1.1M binários a 1 ano).
→ Em `run_case._solve_pwl_per_house` usa-se **decomposição two-stage por casa**:
  1. Stage-1: resolve o modelo base (sem degradação) → trajetória de SoC `E[h,·]`.
  2. Mapeia cada timestep ao bin de SoC → `λ_t` (parâmetro, não variável).
  3. Stage-2: resolve o base + termo linear `Σ λ_t·(P_ch+P_dis)·dt`. **Sem binários PWL.**
O monolítico PWL nunca é construído (o objeto `mh` na `run_case` serve só de metadados).

## 4. Cenários (building8)

Todos herdam de `_b8_base.yaml` via `extends`. Alocação de PV (`alpha`) por estratégia:
equal / weighted / consumption_instant / consumption_mean / optimal.

- pv01–04: heurísticas, com export. pv03 (consumption_instant) é a melhor heurística (~2% do ótimo).
- pv05–08: idem, sem export.
- pv09: optimal + export (limite superior teórico).
- pv10: sem bateria. pv11: sem partilha (bateria genérica). pv12: baseline "nada".
- pv13–14: degradação linear (optimal / weighted).
- pv15–19: **degradação PWL** (equal/weighted/cons_inst/cons_mean/optimal).
- pv20: optimal, sem export.

## 5. Parâmetros

**Baterias (Pylontech reais)** — em `_b8_base.yaml` e `data/battery_values_load_01_to_08.csv`:

| Apt | Módulo | E_max | E_min | E_init | P_ch/dis | custo € | P_contr. |
|-----|--------|-------|-------|--------|----------|---------|----------|
| 1,2,4 | US3000C | 3.55 | 0.18 | 1.78 | 1.78 | 2308 | 2.30 |
| 3,6,8 | US2000C | 2.40 | 0.12 | 1.20 | 1.15 | 1560 | 1.15 |
| 5 | US5000 | 4.80 | 0.24 | 2.40 | 3.45 | 3120 | 3.45 |
| 7 | US5000 | 4.80 | 0.24 | 2.40 | 2.30 | 3120 | 2.30 |

Comuns: `η_ch=η_dis=0.95`, `N_rated_cycles=6000`, `DoD_rated=0.95`, `aging_exponent=1.50`.
Custo ≈ **650 €/kWh** constante entre tipos.

**Tarifa** bi-horária (EDP Verde, c/IVA 23%): ponta 0.24 (08:00–22:00), vazio 0.14, venda 0.06 €/kWh.

**Wöhler λ** (`battery_economics.compute_degradation_cost_per_kwh`):
`N_actual=N_rated·(DoD_rated/DoD_actual)^γ`; `λ=custo/(N_actual·E_usable·(1/η_ch+η_dis))`.
λ global ≈ **0.0569 €/kWh** de throughput AC (igual para todas por o €/kWh ser constante).

## 6. Alterações desta sessão (correção de pontas soltas)

1. **Bins PWL calibrados via Wöhler.** `lambda_by_bin` passou de `[0.08,0.03,0.06]` (manual)
   para `[0.0554,0.0413,0.0185]` em pv15–19. Método: DoD representativo por bin = `1 − midpoint(SoC)`
   → `[0.9,0.5,0.1]`; `λ(DoD)=C·DoD^(γ−1)`, recupera o λ global em `DoD_actual`. Perfil
   **monótono** (descarga profunda = mais cara); uma curva Wöhler pura de DoD não tem penalização
   de SoC alto. Função reprodutível: `battery_economics.compute_pwl_lambda_by_bin`. Default do
   modelo (`_DEFAULT_LAMBDA_BY_BIN`) também atualizado.

2. **Sync de configs verificado — estava tudo consistente** (pv20 em runset+plotset, P_contracted
   do pv12 correto, `battery_values.csv` = `_b8_base.yaml`, 20 cenários cobertos pelos runsets).

3. **Docs**: não existem `.tex`/`.md` na pasta (os das sessões antigas nunca foram gravados aqui).

4. **Rolling horizon implementado** — módulo novo `src/batEnv/utils/rolling_horizon.py`
   (`solve_rolling_horizon`, `resolve_window_step`). Resolve por janelas sobrepostas
   (`window` = commit + look-ahead, commita `step`, transporta SoC). Cíclico global só na última
   janela. Devolve um shell de horizonte completo com valores populados (não resolvido) para a
   extração a jusante funcionar igual. **Opt-in** via `model.rolling_horizon: {enabled, window, step}`.
   Ligado às duas stages do `_solve_pwl_per_house`. Adicionado ao `case_schema` e registado no `meta.yaml`.

5. **Ativado no `runset_pwl_sweep.yaml`** só para 6 meses e 1 ano (override
   `rolling_horizon {enabled:true, window:1344, step:672}`); 1 mês e 3 meses ficam **exatos** (baseline).

6. **Bug crítico corrigido — threading.** O `_solve_pwl_per_house` corria as 8 casas em
   `ThreadPoolExecutor`, mas o `appsi_highs` captura `stdout/stderr` globalmente (Pyomo tee), o que
   **não é thread-safe** → solves corrompidos (falsas "infeasibilities") e deadlock de I/O no Spyder.
   Mudado para **sequencial** (`allow_parallel = False` em `run_case.py`, dentro de `_solve_pwl_per_house`).

7. **Validação do rolling** (pv15, 3 meses, exato vs rolling): throughput **idêntico** (1301.843 kWh),
   custo de degradação 55.13 € vs 55.12 € (**−0.015%**). Rolling praticamente exato → confiável a 6m/1a.

## 7. Estado atual

- `runset_parent.yaml`: **bateria completa ativa** (`alfa_deg_sweep`, `full_horizon_sweep`,
  `pwl_sweep` = true; `pwl_validation` = false). Pronto para correr tudo.
- `runset_pwl_validation.yaml` existe (pv15 3m com rolling → `results/pwl_validation/`), desativado.
- Próximo passo do utilizador: correr `run_experiment.py` para a bateria completa e depois analisar
  (custo anual por cenário, SSR, impacto da degradação e do rolling).

## 8. Quirks do ambiente (IMPORTANTE para a próxima sessão)

- **Editar sempre nos caminhos reais do Windows** (ferramentas Read/Edit). O sandbox Linux do
  assistente está dessincronizado e **trunca** ficheiros existentes → `py_compile`/greps no sandbox
  são pouco fiáveis para ficheiros já existentes (ficheiros novos sincronizam bem).
- A ferramenta **Glob não indexa a pasta `results/`**, mas **Read lê-a** normalmente.
- **Manter o PWL sequencial** (não reativar threads sem resolver a captura de stdout do solver).
- Overrides de runset usam `deep_merge` **recursivo** (`scripts/_common.py`), por isso um override de
  `model` acrescenta chaves sem apagar as existentes (ex.: `rolling_horizon` não apaga `battery_degradation_pwl`).

## 9. Itens em aberto / futuro

- Correr a bateria completa e fazer a análise comparativa final.
- pv19 (optimal-alpha PWL): a stage-1 monolítica de alocação de PV não usa rolling, mas é o mesmo
  modelo que o pv09 (que fecha bem), por isso não é estrangulamento.
- Otimização menor: a stage-1 com rolling constrói um shell de horizonte completo desnecessário.
- O utilitário rolling é genérico (fixed-alpha) e podia ligar-se ao caminho não-PWL, se preciso
  (não é, esses fecham exato a 1 ano).
- Modelo de **bateria comunitária** (vs individual) — pergunta antiga nunca desenvolvida.
- Regenerar os docs LaTeX com os parâmetros atuais (Pylontech, 6000 ciclos), se forem precisos.

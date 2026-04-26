# cases/

Configuração das simulações em YAML. Hierarquia em três camadas:

```
cases/
├── runset_parent.yaml          ← entry point para correr (run_experiment.py)
├── plotset_parent.yaml         ← entry point para gráficos (render_results.py)
│
├── runsets/
│   └── runset_*.yaml           ← um conjunto de cenários a correr
├── plotsets/
│   └── plotset_*.yaml          ← um conjunto de cenários a plottar
│
└── scenarios/
    └── <building>/
        ├── _<building>_base.yaml   ← template partilhado (não corre sozinho)
        └── <case>.yaml             ← cenário concreto, faz `extends:` ao base
```

## Os 3 tipos de "set" YAML

### Parent (lista de filhos)

```yaml
runset:
  - runsets/runset_pv_building8.yaml
  - runsets/runset_outro_estudo.yaml
enabled:
  pv_building8_share: true     # nome do `runset:` interno do filho
defaults:
  outputs_dir: results
  ...
```

A estrutura do `plotset_parent.yaml` é idêntica, trocando `runset` por `plotset`.

### Single runset

```yaml
runset: nome_do_runset
cases_base_dir: cases/scenarios/building8
cases_glob: "b8_*.yaml"          # ou cases: [a.yaml, b.yaml, ...]
enabled:
  b8_pv03_consumption_instant_export: false   # só para desligar específicos
defaults:
  outputs_dir: results/pv_building8
  tee: false
  solver: { name: highs, options: {} }
  time:
    horizon: 2688
    dt_hours: 0.25
    start: 2025-01-01 00:00
```

`cases_glob` é o caminho recomendado: o padrão é uma única fonte de verdade
partilhada entre runset e plotset. Ficheiros começados por `_` são templates
e são automaticamente excluídos pelo padrão `b8_*.yaml`.

### Single plotset

Igual ao single runset mas com `plotset:` em vez de `runset:` e uma secção
`plots:`:

```yaml
plotset: nome_do_plotset
cases_base_dir: cases/scenarios/building8
outputs_dir: results/pv_building8
cases_glob: "b8_*.yaml"
plots:
  per_case: true
  comparisons: true
```

## O mecanismo `extends:`

Os case YAMLs concretos podem herdar de um base via `extends:`:

```yaml
# b8_pv01_equal_export.yaml
extends: _b8_base.yaml      # path relativo ao próprio YAML
case: b8_pv01_equal_export
grid:
  allow_export: true
sharing:
  mode: fixed_alpha
  alpha: { Apt1: 0.125, Apt2: 0.125, ... }
```

O loader carrega o base, faz deep-merge recursivo com o filho (filho ganha
em conflito), e remove a chave `extends` do resultado. `extends:` aceita
string ou lista (merge sequencial, último ganha). Cadeias cíclicas levantam
`ValueError`.

## Validação

Todos os YAMLs são validados via `jsonschema` antes de serem usados:

| YAML | Schema | Validador |
|---|---|---|
| `cases/scenarios/.../<case>.yaml` | `CASE_SCHEMA` | `validate_case_cfg_schema` |
| `cases/runsets/...` e `cases/runset_parent.yaml` | `RUNSET_SCHEMA` | `validate_runset_cfg` |
| `cases/plotsets/...` e `cases/plotset_parent.yaml` | `PLOTSET_SCHEMA` | `validate_plotset_cfg` |

Os schemas são restritivos em secções fixas (`battery`, `fallback`, `time`,
`grid`, `defaults`, `solver`) — typos em chaves obrigatórias são apanhados
antes da simulação correr. Se `jsonschema` não estiver instalado, a
validação é saltada com um warning.

## Convenções de nomes

- Templates partilhados começam com `_` (ex: `_b8_base.yaml`).
- Cenários concretos seguem `<building>_<descritor>.yaml`.
- Os nomes em `enabled:` referem-se ao campo `case:` (cenários) ou
  `runset:`/`plotset:` (orquestração) **dentro** do YAML, não ao nome do
  ficheiro.

## Como adicionar um cenário novo

1. Criar `cases/scenarios/<building>/<novo>.yaml` com `extends: _<building>_base.yaml`
   + apenas as overrides necessárias.
2. O `cases_glob` no runset/plotset apanha-o automaticamente — nada mais a fazer.
3. Para correr: `python scripts/run_experiment.py --runset cases/runset_parent.yaml`
4. Para plottar (após correr): `python scripts/render_results.py --plotset cases/plotset_parent.yaml`

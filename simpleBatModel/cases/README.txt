README — cases replacement (PV Study v2)
======================================

Este ZIP foi preparado para SUBSTITUIR a pasta 'cases/' do projeto.

Conteúdo:
- cases/pv_study_v2/*.yaml  (8 casos, matriz 4×2: partilha × export ON/OFF)
- cases/runset.yaml         (aponta para pv_study_v2)
- cases/plotset.yaml        (aponta para pv_study_v2)
- cases/COMPARACAO_ENTRE_CASOS.txt (guia e comparação dos casos)

Notas:
- Os ficheiros de dados referenciados assumem o layout original do repo:
    data/load_1.csv ... data/load_4.csv, data/pv_gen.csv
- Export OFF exige o código com P_curt (curtailment) e grid.allow_export implementados.

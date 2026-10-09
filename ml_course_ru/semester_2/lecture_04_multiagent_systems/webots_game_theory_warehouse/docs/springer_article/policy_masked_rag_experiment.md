# Policy-Masked RAG/Embedding Experiment

## Цель

Проверить, дает ли retrieval по embedding rules измеримый эффект, если прямое эксплуатационное правило не передается в prompt. В основной LM-9 сцене `scene_profile` уже содержит явную инструкцию про инспекционный трафик, дозирование выпуска и предпочтение обходов, поэтому LLM без RAG может принять похожее решение. Новый эксперимент скрывает эту расшифровку.

## Изменение сценария

Добавлено семейство `policy_masked` в `statistical_experiment.py`.

- В prompt передаются только коды `SOP-C7` и `DOCK-AMBER`.
- В prompt не передаются прямые слова `bypass`, `maintenance`, `inspection`, `disrupted_edges` и явный флаг центрального конфликта.
- В базе правил добавлены документы `sop_c7_center_policy` и `dock_amber_metering_policy`.
- Число грузов увеличено до 15-18, роботов 6-8, так что каждый робот в среднем выполняет несколько доставок.
- Сравнение выполняется на 30 seed для deterministic replay и на 10 seed для live Ollama pilot.

## Основной результат, deterministic replay

| Режим | n | Makespan, с | CI95 | Bypass | Success |
|---|---:|---:|---:|---:|---:|
| Keyword parser | 30 | 1932.3 ± 237.5 | 85.0 | 0.0 | 0.833 |
| LLM no RAG replay | 30 | 1882.9 ± 254.1 | 90.9 | 0.0 | 0.867 |
| Embedding rule | 30 | 1517.3 ± 240.0 | 85.9 | 11.1 | 0.967 |
| RAG-LLM replay | 30 | 1315.4 ± 159.2 | 57.0 | 12.9 | 1.000 |
| Static rule | 30 | 1255.8 ± 160.3 | 57.4 | 11.1 | 1.000 |

Относительно `LLM no RAG replay`:

- `embedding_rule` улучшает makespan на 365.6 с.
- `RAG-LLM replay` улучшает makespan на 567.5 с.
- paired sign test для `LLM no RAG replay` против `RAG-LLM replay`: p = 1.86e-9.

Вывод: embedding/RAG дает измеримый эффект, когда правило не раскрыто напрямую в prompt и должно быть извлечено из локальной базы.

## Live Ollama Pilot

| Режим | n | Makespan, с | CI95 | Bypass | Latency, с | Success |
|---|---:|---:|---:|---:|---:|---:|
| LLM no RAG live | 10 | 1439.7 ± 398.7 | 247.1 | 8.4 | 8.00 | 0.900 |
| RAG-LLM live | 10 | 1331.2 ± 139.8 | 86.7 | 10.8 | 8.07 | 1.000 |

Live Ollama сохраняет знак эффекта: RAG уменьшает makespan на 108.5 с и чаще выбирает обходные маршруты. Однако на 10 seed paired sign test дает p = 1.0, поэтому это следует трактовать как pilot, а не как окончательное статистическое доказательство.

## Устойчивость к ошибкам retrieval

При искусственной подмене каждого retrieved-документа с вероятностью 0.5:

| Режим | n | Makespan, с | CI95 | Wait, с | Success |
|---|---:|---:|---:|---:|---:|
| Embedding rule, corrupted | 30 | 1611.3 ± 297.9 | 106.6 | 576.1 | 0.933 |
| RAG-LLM replay, corrupted | 30 | 1588.0 ± 365.2 | 130.7 | 733.3 | 0.967 |

Ошибки retrieval заметно ухудшают результат: RAG-LLM replay деградирует с 1315.4 до 1588.0 с. Это подтверждает, что retrieved rules причинно влияют на стратегию, а не просто логируются для объяснимости.

## Ответ на вопрос о правилах

Текущих правил было почти достаточно для событий вида `audit vehicle`, `temporary human check` и `occupied middle aisle`, но для строгого masked-сценария нужны SOP-документы, иначе векторная БД не сможет расшифровать коды `SOP-C7` и `DOCK-AMBER`. Поэтому добавлены два правила:

- `sop_c7_center_policy`: связывает SOP-C7/amber с capacity-reduced center crossing и предпочтением outer corridors.
- `dock_amber_metering_policy`: связывает DOCK-AMBER/picker slot pressure с дозированием выпуска роботов.

Именно эта конфигурация лучше демонстрирует роль vector DB: новые эксплуатационные правила добавляются в `warehouse_rules.json`, а Python-логика назначения и маршрутизации не меняется.

## Файлы

- `controllers/warehouse_supervisor/statistical_experiment.py`
- `controllers/warehouse_supervisor/warehouse_rules.json`
- `experiment_results/policy_masked_sop_embedding/statistical_experiment_results.json`
- `experiment_results/policy_masked_retrieval_error_50/statistical_experiment_results.json`
- `experiment_results/policy_masked_live/statistical_experiment_results.json`

# Webots: многоагентная складская доставка с теорией игр и локальной LLM

Этот проект демонстрирует многоагентную транспортную систему для склада в Webots. Несколько мобильных роботов перевозят грузы между зонами склада, договариваются о задачах через сетевые сообщения, выбирают маршруты с помощью формальных моделей теории игр и локально используют LLM как стратегический советник.

Сценарий сделан самодостаточным: роботы описаны локальным `PROTO WarehouseRobot`, поэтому мир не зависит от наличия стандартных PROTO из `/Applications/Webots.app/projects`. Координаты соответствуют Webots ENU: горизонтальная плоскость `XY`, вертикаль `Z`, положительный `X` направлен на восток, положительный `Y` на север.

## Что происходит на сцене

Основная сцена теперь настроена как `large_maintenance_9_tasks_8_robots`: на складе есть восемь роботов `ROBOT_0` ... `ROBOT_7`, девять грузов и несколько доков. Все задачи `A/B/C`, `A2/B2/C2`, `A3/B3/C3` открыты с начала симуляции. Поток регулируется не искусственными волнами, а стратегическим шлюзом: LLM/RAG возвращает, каких роботов выпустить к новым грузам сейчас, а каких оставить ждать освобождения коридора, pickup-зоны или dropoff-зоны.

Три группы грузов повторяют одни и те же транспортные направления:

- `A`: забрать груз в юго-западной зоне склада (`X < 0`, `Y < 0`) и доставить в северо-восточный док (`X > 0`, `Y > 0`).
- `B`: забрать груз в северо-западной зоне (`X < 0`, `Y > 0`) и доставить в юго-восточный док (`X > 0`, `Y < 0`).
- `C`: забрать груз у южного входа центрального коридора (`Y < 0`) и доставить к северному доку (`Y > 0`).

Склад содержит стеллажи, стены, центральный перекресток, объездные коридоры, зоны pickup/dropoff и визуальные маркеры маршрутов. Центральный перекресток намеренно конфликтный: если все роботы выбирают кратчайшие пути, возникают заторы и взаимные уступания. Центральные ребра в основной сцене не закрываются жестко: LLM повышает штраф заторов, но `blocked_edges` обычно остается пустым, поэтому роботы могут ехать по синим центральным линиям.

Текущая Webots-сцена увеличена по горизонтальным координатам в `3.4` раза относительно первого компактного прототипа. Площадь пола стала больше в `11.56` раза, а общие координаты вынесены в `controllers/warehouse_supervisor/warehouse_geometry.py`, чтобы мир, планировщик, dry-run и эксперимент использовали один масштаб.

## Главная идея архитектуры

Система построена как распределенная многоагентная система с сетевым координатором. Supervisor не водит роботов напрямую по сцене. Каждый робот имеет собственный контроллер, сам принимает локальные решения движения, обменивается сообщениями и сообщает о статусе.

```text
                 +-------------------------+
                 | WarehouseSupervisor     |
                 | network coordinator     |
                 |                         |
                 | - task board            |
                 | - LLM/RAG strategy      |
                 | - Contract Net auction  |
                 | - assignment broadcast  |
                 +-----------+-------------+
                             |
              Webots Emitter/Receiver, channel 1
                             |
      +----------------------+----------------------+
      |                      |                      |
+-----+------+        +------+------+        +------+-----+
| Robot R0   |        | Robot R1    |        | Robot R2/R3 |
| controller |        | controller  |        | controllers |
|            |        |             |        |             |
| - status   |<------>| - peer data |<------>| - status    |
| - A*       |        | - traffic   |        | - A*        |
| - pickup   |        | - collision |        | - done      |
+------------+        +-------------+        +-------------+
```

Supervisor координирует сеть, но не является единственным центром движения. Его роль похожа на диспетчера склада: он публикует стратегию, доску задач и назначения. Исполнение остается на стороне роботов.

## Трехуровневая модель принятия решений

Проект разделяет принятие решений на три уровня.

### 1. Стратегический уровень: RAG + локальная LLM

Файл: `controllers/warehouse_supervisor/llm_strategy.py`

LLM получает компактное состояние склада и несколько правил из локальной векторной базы. На выходе она возвращает не команды моторам, а параметры стратегии:

- `deadline_weight` — насколько сильно учитывать дедлайны;
- `travel_weight` — насколько штрафовать длинные маршруты;
- `congestion_weight` — насколько избегать загруженных коридоров;
- `battery_weight` — насколько учитывать заряд роботов;
- `safety_margin` — запас безопасности для локальных правил движения;
- `blocked_edges` — ребра маршрутного графа, которые стратегически лучше не использовать;
- `go_agents` — роботы, которым сейчас можно участвовать в новом аукционе;
- `hold_agents` — свободные роботы, которым лучше ждать освобождения зоны или коридора;
- `max_active_tasks` — ограничение числа одновременно исполняемых доставок;
- `max_new_assignments` — размер следующей пачки назначений Contract Net;
- `traffic_rule` и `note` — текстовая политика и пояснение.

LLM здесь не заменяет формальную модель. Она играет роль адаптивного стратегического слоя: меняет веса и ограничения, а конкретные назначения, маршруты и движение остаются проверяемыми алгоритмами.

По умолчанию запрос к Ollama выполняется асинхронно: supervisor сразу публикует `fallback_strategy`, продолжает Webots-loop, запускает RAG+LLM в фоновом worker-потоке и применяет новую стратегию только после готового ответа. Если Ollama недоступна, вернула невалидный JSON или заданный timeout истек, используется `fallback_strategy`. Поэтому сценарий остается работоспособным без LLM и не "замораживает" симуляцию во время ожидания модели.

### Контракт данных RAG и LLM

Стратегический pipeline запускается в `ask_llm_strategy` внутри `llm_strategy.py`.

```text
tasks + agent snapshots + time
        |
        v
compact JSON state
        |
        v
SQLite vector DB query
        |
        v
retrieved warehouse rules
        |
        v
LLM prompt for Ollama
        |
        v
raw JSON response
        |
        v
validated Strategy dataclass
        |
        v
Contract Net + congestion game + robot controllers
```

#### Какие правила используются

Правила лежат в `controllers/warehouse_supervisor/warehouse_rules.json`. При старте `LocalVectorStore` читает этот JSON, строит embedding-векторы и сохраняет их в `warehouse_rules.sqlite`.

Текущие документы базы:

- `traffic_right_hand`: дедлайн-приоритет, правило правой стороны, приоритет загруженного робота.
- `collision_static`: запрет сегментов, пересекающих стеллажи, стены, закрытые проходы и другие статические препятствия.
- `collision_dynamic`: другие роботы рассматриваются как движущиеся safety disks.
- `three_level_control`: разделение на стратегический, тактический и операционный уровни.
- `congestion_policy`: центральный перекресток короткий, но общий; при перегрузке надо повышать штраф заторов.
- `large_scene_policy`: в большом складе ожидание у центра может быть дороже лишних метров обхода.
- `maintenance_center_policy`: при инспекции/ремонте центральных ребер надо распределять поток по обходам.
- `dispatch_metering_policy`: при большом числе открытых заказов выпускать не всех роботов сразу, а возвращать `go_agents`, `hold_agents`, `max_active_tasks`, `max_new_assignments`.
- `warehouse_axes`: координаты Webots ENU: восток `+X`, север `+Y`, высота `+Z`.

#### Что входит в векторную СУБД

На вход индексации идут документы вида:

```json
{
  "id": "collision_dynamic",
  "text": "Dynamic collision avoidance treats other robots as moving safety disks..."
}
```

`vector_memory.py` токенизирует `text`, строит PyTorch embedding фиксированной размерности `192`, нормализует его и сохраняет в SQLite:

```text
documents(id, text, embedding, dimension, encoder, updated_at)
metadata(key, value)
```

`embedding` хранится как `float32` BLOB. Это локальная учебная vector DB: без отдельного сервера, но с явным persistent index.

#### Что подается на вход vector search

На вход поиска подается компактное состояние склада, сериализованное в JSON-строку. Его строит `_compact_state`:

```json
{
  "time": 42.0,
  "tasks": [
    {
      "id": "A",
      "status": "open",
      "deadline_in": 38.0,
      "reward": 12.0,
      "assigned_agent": null,
      "route": null
    }
  ],
  "agents": [
    {
      "id": "R0",
      "status": "idle",
      "battery": 0.91,
      "task": null
    }
  ],
  "possible_edges": [
    "west_center",
    "east_center",
    "south_center",
    "north_center",
    "south_bypass",
    "north_bypass",
    "east_bypass",
    "west_bypass"
  ],
  "dispatch_context": {
    "open_orders": 9,
    "idle_agents": 8,
    "policy_need": "choose go_agents hold_agents max_active_tasks max_new_assignments for pickup dropoff aisle capacity"
  }
}
```

#### Что выходит из векторной СУБД

На выходе `LocalVectorStore.query` возвращает top-k объектов `RetrievedRule`:

```json
[
  {
    "rule_id": "collision_dynamic",
    "score": 0.18,
    "text": "Dynamic collision avoidance treats other robots as moving safety disks..."
  },
  {
    "rule_id": "three_level_control",
    "score": 0.09,
    "text": "The three-level control model separates strategic policy..."
  }
]
```

Эти правила не исполняются напрямую. Они добавляются в prompt как policy memory для LLM.

#### Что подается на вход LLM

На вход Ollama отправляется prompt из трех частей:

1. Инструкция вернуть только компактный JSON с разрешенными ключами.
2. Retrieved rules из векторной СУБД.
3. Compact state текущего склада.

Упрощенная форма prompt:

```text
You are a strategic dispatcher for a multi-agent warehouse.
Return only compact JSON with keys:
deadline_weight, travel_weight, congestion_weight, battery_weight,
safety_margin, traffic_rule, blocked_edges, go_agents, hold_agents,
max_active_tasks, max_new_assignments, note.
...
Retrieved rules:
- collision_dynamic (score=0.18): ...
- three_level_control (score=0.09): ...

State:
{"time": 42.0, "tasks": [...], "agents": [...], "possible_edges": [...]}
```

#### Что LLM должна вернуть

LLM должна вернуть JSON-объект:

```json
{
  "deadline_weight": 1.6,
  "travel_weight": 1.0,
  "congestion_weight": 2.4,
  "battery_weight": 0.8,
  "safety_margin": 0.7,
  "traffic_rule": "deadline_priority_right_hand",
  "blocked_edges": [],
  "go_agents": ["R4", "R5", "R6"],
  "hold_agents": ["R0", "R1", "R2", "R3", "R7"],
  "max_active_tasks": 3,
  "max_new_assignments": 3,
  "note": "Prefer balanced routing because the center is congested."
}
```

Ответ проходит валидацию:

- веса ограничиваются диапазоном `0.3..4.0`;
- `safety_margin` ограничивается диапазоном `0.45..1.2`;
- `blocked_edges` фильтруется по списку известных ребер;
- `go_agents` и `hold_agents` фильтруются по реально существующим роботам;
- `max_active_tasks` и `max_new_assignments` ограничиваются малым безопасным диапазоном;
- если JSON невалиден, используется `fallback_strategy`.

После валидации создается `Strategy`. Именно `Strategy`, а не текст LLM, попадает в формальные модели:

- `deadline_weight` влияет на срочность задач в Contract Net;
- `travel_weight` меняет штраф длины маршрута;
- `congestion_weight` меняет штраф общих коридоров в congestion game;
- `battery_weight` влияет на штраф роботов с низким зарядом;
- `safety_margin` передается в локальные правила движения;
- `blocked_edges` добавляет большой штраф выбранным ребрам.
- `go_agents` и `hold_agents` становятся диспетчерским шлюзом перед Contract Net.
- `max_active_tasks` и `max_new_assignments` не дают открыть слишком много одновременных рейсов.

Таким образом LLM не управляет роботами напрямую. Она меняет параметры математической модели, а все назначения, маршруты и движения остаются в детерминированном коде.

### 2. Тактический уровень: теория игр

Файл: `controllers/warehouse_supervisor/game_models.py`

На этом уровне решаются две задачи.

Сначала работает `Contract Net Protocol`: свободные роботы делают ставки за открытые задачи. Ставка учитывает награду, срочность, расстояние до pickup-точки и штраф за низкий заряд:

```text
bid = reward + urgency - pickup_cost - battery_penalty
```

Затем выбранные роботы участвуют в `Congestion Game`: каждый агент выбирает маршрут, а стоимость маршрута растет, если тот же коридор используют другие агенты. Функция `choose_route_nash_equilibrium` реализует iterated best response: агенты по очереди выбирают лучший ответ на маршруты остальных, пока профиль не стабилизируется.

```text
route_cost = path_length * travel_weight
           + congestion * congestion_weight
           + blocked_edge_penalty
```

Тактический уровень возвращает `Assignment`: какой робот выполняет какую задачу и каким маршрутом.

### 3. Операционный уровень: контроллер каждого робота

Файл: `controllers/warehouse_robot_agent/warehouse_robot_agent.py`

Каждый робот:

- принимает `task_board`, `strategy` и `assignment`;
- публикует свой `status` для supervisor и других роботов;
- хранит локальное состояние соседних роботов;
- строит путь до pickup и dropoff через `planner.py`;
- проверяет статические и динамические коллизии;
- применяет правила уступания из `traffic_rules.py`;
- переносит назначенный груз;
- отправляет события `pickup`, `done`, `replan`, `assignment_ack`.

Роботы обмениваются сообщениями через Webots `Emitter`/`Receiver` на общем канале. Это не физическая радиосеть, а учебная модель коммуникации MAS.

## Компоненты проекта

```text
webots_game_theory_warehouse/
  worlds/
    game_theory_warehouse.wbt
      Сцена склада: стены, стеллажи, доки, девять грузов, supervisor, восемь роботов.

  protos/
    WarehouseRobot.proto
      Локальный мобильный складской робот: корпус, колеса, платформа,
      Receiver и Emitter. По умолчанию запускает warehouse_robot_agent.

  controllers/
    warehouse_robot_agent/
      warehouse_robot_agent.py
        Индивидуальный контроллер робота. Отвечает за движение, локальное
        планирование, объезд, перенос груза и обмен peer-сообщениями.

    warehouse_supervisor/
      warehouse_supervisor.py
        Сетевой координатор. Публикует task board, стратегию и назначения,
        принимает статусы и события роботов. Не перемещает роботов напрямую.

      game_models.py
        Формальные модели: Task, Route, Strategy, Assignment,
        Contract Net Protocol, congestion game, best-response dynamics.

      llm_strategy.py
        Стратегический слой RAG + Ollama. Нормализует JSON-ответ LLM и
        защищает формальную модель от невалидных значений.

      vector_memory.py
        Локальная векторная СУБД правил на SQLite. Строит PyTorch embeddings,
        хранит их в warehouse_rules.sqlite и ищет top-k по cosine similarity.

      warehouse_rules.json
        Набор правил склада для RAG: правила движения, коллизии, оси,
        трехуровневая архитектура, политика заторов.

      warehouse_rules.sqlite
        Автоматически создаваемый SQLite-индекс с embedding-векторами.

      planner.py
        Occupancy grid, A*, проверка точек и сегментов на статические и
        динамические препятствия.

      traffic_rules.py
        Локальные правила уступания: дедлайн-приоритет, приоритет загруженного
        робота, правило правой стороны, safety margin.

      scenario.py
        Описание задач и кандидатных маршрутов.

      dry_run.py
        Headless-проверка логики без Webots GUI.

      inspect_vector_db.py
        Просмотр содержимого векторной базы и результатов поиска.

    passive_robot/
      passive_robot.py
        Legacy-заглушка. В текущем мире не используется.

  docs/
    SCENE_AND_ARCHITECTURE.md
      Подробное описание сцены, моделей и UML-диаграмм.

    LLM_VALUE_EXPERIMENT.md
      Вычислительный эксперимент: сравнение KPI сценариев с LLM-стратегией
      и без LLM.
```

## Имеет ли смысл использовать LLM в этой задаче

Для самой транспортной задачи LLM не обязательна. Назначение задач, выбор маршрутов, объезд препятствий и предотвращение столкновений можно полностью решить классическими методами: аукционами, теорией игр, A*, правилами движения и конечными автоматами.

В этом примере LLM имеет смысл как учебный стратегический слой:

- показывает, как языковая модель может быть встроена в MAS без прямого управления моторами;
- демонстрирует RAG над локальной базой правил;
- позволяет менять стратегические веса и политики без переписывания формальных алгоритмов;
- дает наблюдаемый pipeline `state -> retrieved rules -> LLM strategy -> game models -> robot controllers`;
- хорошо иллюстрирует границу ответственности: LLM советует, формальная модель проверяет и исполняет.

LLM не стоит использовать здесь для:

- прямого построения траектории;
- обхода коллизий в реальном времени;
- выбора скорости колес;
- единственного источника правил безопасности;
- критически важного hard real-time управления.

Итог: LLM полезна для демонстрации гибридной архитектуры, но ядро системы должно оставаться формальным и детерминированно проверяемым. Именно так проект и устроен: при отказе LLM включается fallback, и склад продолжает работать.

## Apple Silicon Ollama

На Mac M1/M2/M3 важно использовать нативную `arm64`-версию Ollama. Если команда `ollama` указывает на `/usr/local/bin/ollama`, это часто Intel/Rosetta-сборка. Для Apple Silicon предпочтителен путь `/opt/homebrew/bin/ollama`.

Проверка:

```bash
which ollama
file $(which ollama)
ollama ps
```

Хороший результат:

```text
/opt/homebrew/bin/ollama
Mach-O 64-bit executable arm64
PROCESSOR 100% GPU
```

Установка через Apple Silicon Homebrew:

```bash
/opt/homebrew/bin/brew install ollama
/opt/homebrew/bin/brew services start ollama
```

Если раньше была установлена Intel-версия через `/usr/local`, ее можно убрать из PATH-приоритета:

```bash
/usr/local/bin/brew services stop ollama
/usr/local/bin/brew unlink ollama
```

После переключения в этой среде `qwen2.5:0.5b` на прогретой GPU-модели выполняет полный стратегический запрос примерно за несколько секунд. Первый cold-start может быть дольше, потому что модель загружается в память GPU.

## Запуск с локальной Ollama

Модель по умолчанию:

```bash
ollama pull qwen2.5:0.5b
```

Проверка:

```bash
ollama run qwen2.5:0.5b "Return JSON: {\"deadline_weight\": 2}"
```

Проект по умолчанию использует `qwen2.5:0.5b`. Другую модель можно задать переменной окружения:

```bash
export OLLAMA_MODEL=smollm2:135m
```

`smollm2:135m` меньше, но в тестах хуже держит строгий JSON для полного prompt. Поэтому текущий рекомендуемый вариант для этого проекта — `qwen2.5:0.5b` на нативной Apple Silicon Ollama.

По умолчанию supervisor не останавливает симуляцию ради Ollama. Он стартует с fallback-стратегией, отправляет LLM-запрос в фоновый поток и продолжает аукционы, маршрутизацию и обмен сообщениями на последней валидной стратегии:

```bash
export MAS_ASYNC_LLM=1
export MAS_ASYNC_LLM_TIMEOUT=35
```

`MAS_ASYNC_LLM_TIMEOUT` задает верхнюю границу фонового запроса в секундах. Следующий стратегический запрос не ставится в очередь, пока предыдущий не завершился: устаревшая стратегия хуже, чем спокойная работа на последней валидной.

Для отладки старого блокирующего режима можно явно выключить async. В этом случае Webots-loop будет ждать ответ Ollama, а `MAS_LLM_TIMEOUT=0` означает ждать без короткого лимита:

```bash
export MAS_ASYNC_LLM=0
export MAS_LLM_TIMEOUT=0
```

Чтобы в blocking-режиме не ждать слишком долго, можно задать числовой предел:

```bash
export MAS_ASYNC_LLM=0
export MAS_LLM_TIMEOUT=10
```

Для демонстрации fallback-режима в blocking-режиме:

```bash
export MAS_ASYNC_LLM=0
export MAS_LLM_TIMEOUT=1
```

## Запуск Webots

1. Открыть Webots.
2. Открыть мир:

```text
ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/worlds/game_theory_warehouse.wbt
```

3. Нажать Run.
4. Смотреть консоль:

- supervisor печатает `[strategy]`, `[strategy-async]`, `[dispatch-gate]`, `[dispatch-wait]`, `[auction]`, `[pickup]`, `[done]`, `[success]`;
- LLM-слой печатает `[vector-db]`, `[vector-query]`, `[llm-request]`, `[llm-response]`, `[llm-fallback]`;
- роботы печатают assignment/pickup/done/replan и отправляют статусы.

## Проверка без Webots GUI

Быстрая headless-проверка формальных моделей и логики движения:

```bash
python3 ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/dry_run.py
```

Ожидаемый финал:

```text
SUCCESS completed=A,B,C,A2,B2,C2,A3,B3,C3
```

## Эксперимент LLM vs fallback

Сравнение стратегий на нескольких больших сценах запускается так:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache \
MAS_VECTOR_DEBUG=0 \
OLLAMA_MODEL=qwen2.5:0.5b \
/Users/hiber/miniforge3/bin/python3 \
  ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/llm_value_experiment.py \
  --timeout 18 --dt 1.5
```

В вычислительном эксперименте LLM дала преимущество не в базовой сцене, а в конфигурации `large_maintenance_9_tasks_8_robots`, где большой склад, 8 роботов, 9 открытых задач и несколько альтернативных маршрутов делают стратегическую настройку весов полезной. В текущей Webots-сцене центральные ребра не закрываются жестко: `blocked_edges=[]`, LLM повышает `congestion_weight`, а Contract Net ограничивает одновременные рейсы вместимостью семейств зон `A/B/C`. Подробный отчет: `docs/LLM_VALUE_EXPERIMENT.md`.

## Логи LLM и векторной СУБД

Обычный стратегический цикл выглядит так:

```text
[vector-db] path=... db=.../warehouse_rules.sqlite backend=sqlite+pytorch encoder=torch_hashing_bow_v1 documents=9 dimension=192 ids=...
[vector-query] query="..." top=collision_static:0.43, congestion_policy:0.31
[llm-request] model=qwen2.5:0.5b url=http://127.0.0.1:11434/api/generate timeout=wait ...
[llm-response] {"deadline_weight": ..., "retrieved_rules": ...}
[strategy] {"deadline_weight": ..., "congestion_weight": ...}
```

Если Ollama недоступна или ответ невалиден:

```text
[llm-fallback] reason=...
```

Включить полный prompt и сырой ответ:

```bash
export MAS_LLM_DEBUG=1
```

Отключить подробные логи поиска по векторной СУБД:

```bash
export MAS_VECTOR_DEBUG=0
```

Посмотреть содержимое локальной векторной СУБД:

```bash
python3 ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/inspect_vector_db.py
```

Проверить поиск:

```bash
python3 ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/inspect_vector_db.py \
  --query "robots near intersection should avoid collisions" --top-k 3
```

Показать первые компоненты сохраненных embedding-векторов:

```bash
python3 ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/inspect_vector_db.py --vectors
```

PyTorch установлен в окружение Webots Python. Если на другой машине PyTorch отсутствует, `vector_memory.py` автоматически переключится на `sqlite+python`: SQLite-индекс останется, но embedding и cosine search будут считаться чистым Python.

## Коллизии и объезд

Столкновения предотвращаются на операционном уровне, внутри каждого робота:

- `planner.py` проверяет границы склада, стены, стеллажи и динамические препятствия;
- `WarehouseRobotAgent.peer_obstacles` строит список соседних роботов по сообщениям `status`;
- `TrafficRuleEngine.must_yield` решает, должен ли робот уступить;
- если прямой сегмент небезопасен, робот локально перестраивает путь через A* и отправляет `replan`.

Это не физически точная модель колесной динамики. Робот кинематически обновляет `translation` и `rotation` своего Webots-узла. Для учебного примера MAS это делает логику принятия решений наблюдаемой и устойчивой.

## Стандартные роботы Webots

Для более реалистичной версии можно заменить локальный `WarehouseRobot` на стандартный `youBot`: мобильный робот KUKA с манипулятором и захватом. В этом проекте оставлен самодостаточный робот, потому что в локальной установке Webots стандартные PROTO могут отсутствовать или лежать в другой версии `projects`.

При замене важно сохранить:

- `DEF ROBOT_0` ... `DEF ROBOT_7`;
- поля `translation` и `rotation`;
- сетевые устройства `Receiver` и `Emitter`;
- индивидуальный controller `warehouse_robot_agent`;
- совместимость имен грузов `CARGO_A`, `CARGO_B`, `CARGO_C` и повторов `CARGO_A2` ... `CARGO_C3`.

## Демонстрационные сценарии

Этот раздел можно использовать как план живого показа на занятии. Лучше открыть одновременно Webots-сцену и консоль контроллеров.

### 1. Contract Net Protocol назначает задачи агентам

Что сделать:

```bash
python3 ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/dry_run.py
```

Что показать в логах:

```text
t=000.0 assign R6 -> C via C_center
t=000.0 assign R4 -> A via A_south_bypass
t=000.0 assign R5 -> B via B_center
```

Как объяснить: supervisor собирает `AgentSnapshot`, берет открытые `Task` и запускает `contract_net_auction`. Каждый свободный робот получает оценку `bid = reward + urgency - pickup_cost - battery_penalty`; задача назначается агенту с лучшей ставкой.

Где смотреть код: `controllers/warehouse_supervisor/game_models.py`, функция `contract_net_auction`.

### 2. Congestion game меняет выбор маршрута

Что сделать: запустить Webots и посмотреть строки `[auction]` в консоли supervisor. У каждой задачи есть несколько кандидатных маршрутов: центральный и объездные.

Что показать в логах:

```text
[auction] agent=R6 task=C route=C_center
[auction] agent=R4 task=A route=A_south_bypass
[auction] agent=R5 task=B route=B_center
```

Как объяснить: после выбора победителей аукциона система не просто берет кратчайший путь. `choose_route_nash_equilibrium` несколько раз пересчитывает лучшие ответы агентов. Если общий коридор перегружен, растет `congestion * congestion_weight`, и одному из роботов становится выгоднее взять объезд.

Мини-эксперимент: в `controllers/warehouse_supervisor/llm_strategy.py` или через LLM/fallback поднять `congestion_weight` и сравнить выбранные `route_id`. Чем выше вес затора, тем чаще появляются маршруты с `bypass`.

Где смотреть код: `route_cost` и `choose_route_nash_equilibrium` в `game_models.py`.

### 3. LLM влияет только на стратегические веса

Что сделать: запустить Webots с Ollama и посмотреть стратегический цикл.

Что показать в логах:

```text
[llm-request] model=qwen2.5:0.5b url=http://127.0.0.1:11434/api/generate timeout=wait ...
[llm-response] {"blocked_edges": [], "congestion_weight": 3.8, "go_agents": ["R4", "R5", "R6"], ...}
[strategy] {"blocked_edges": [], "max_active_tasks": 3, "max_new_assignments": 3, ...}
```

Как объяснить: LLM не получает доступа к `translation`, `rotation`, моторам или прямому управлению роботами. Она возвращает только объект `Strategy`. Дальше этот объект используют формальные функции стоимости, а движение выполняют индивидуальные контроллеры.

Где смотреть код: `ask_llm_strategy` в `llm_strategy.py` и `route_cost` в `game_models.py`.

### 4. RAG достает правила из локальной базы

Что сделать:

```bash
python3 ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/inspect_vector_db.py \
  --query "robots near intersection should avoid collisions" --top-k 3
```

Что показать в логах Webots:

```text
[vector-db] path=... warehouse_rules.json db=... warehouse_rules.sqlite backend=sqlite+pytorch documents=9 dimension=192 ids=...
[vector-query] query="..." top=collision_dynamic:0.42, traffic_right_hand:0.31, ...
```

Как объяснить: `vector_memory.py` превращает текст правил и текущее состояние склада в PyTorch-векторы, сохраняет embedding в SQLite как BLOB и выбирает top-k правил по cosine similarity. Это локальная векторная СУБД без внешнего сервера: не Chroma/FAISS, но с теми же ключевыми этапами RAG для учебного примера.

Где смотреть код: `LocalVectorStore.query` и `rules_to_prompt` в `vector_memory.py`.

### 5. Supervisor координирует сеть, но не водит роботов

Что сделать: открыть `warehouse_supervisor.py` и `warehouse_robot_agent.py` рядом с Webots.

Что показать в логах:

```text
[strategy] ...
[auction] agent=R5 task=B route=B_center
[ack] agent=R5 task=B route=B_center
[pickup] agent=R5 task=B
[done] agent=R5 task=B route=B_center completed_at=...
```

Как объяснить: supervisor отправляет сообщения `task_board`, `strategy`, `assignment`, но не выполняет покадровое движение робота. Движение, перенос груза и объезд находятся в `warehouse_robot_agent.py`. Это видно по тому, что события `pickup`, `done`, `replan` приходят обратно от роботов.

Где смотреть код: `WarehouseSupervisor.apply_assignment` только отправляет JSON-сообщение, а `WarehouseRobotAgent.step_motion` обновляет состояние конкретного робота.

### 6. Робот локально избегает препятствий и других роботов

Что сделать: во время Webots-запуска смотреть на центральный перекресток и консоль роботов.

Что показать в логах:

```text
[replan] agent=R0 target=[...] detour=...
```

Как объяснить: каждый робот получает статусы соседей, строит `DynamicObstacle`, проверяет следующий сегмент через `segment_is_free`, применяет `TrafficRuleEngine.must_yield`, а при необходимости перестраивает путь через A*. Если пустой робот оказался в зоне разгрузки загруженного, он делает короткий безопасный шаг освобождения через `retreat_from_loaded_peer`, чтобы не превратить уступание в deadlock.

Если хочется сделать эффект заметнее, можно временно уменьшить свободное пространство в мире или добавить роботу конфликтный маршрут через центральный коридор. Тогда чаще появляются остановки и `replan`.

Где смотреть код: `ensure_collision_free_target`, `retreat_from_loaded_peer`, `peer_obstacles`, `step_motion` в `warehouse_robot_agent.py`; `WarehouseLayout.plan_path` и `segment_is_free` в `planner.py`.

### 7. Fallback сохраняет работоспособность без LLM

Что сделать: запустить Webots или dry-run с намеренно недоступной моделью.

```bash
export OLLAMA_MODEL=missing-model
export MAS_LLM_TIMEOUT=1
```

Что показать в логах:

```text
[llm-request] model=missing-model ...
[llm-fallback] reason=...
[llm-response] {"note": "fallback: late tasks=..., busy agents=...", ...}
[strategy] {"deadline_weight": ..., "congestion_weight": ..., ...}
```

Как объяснить: отказ LLM не останавливает склад. `fallback_strategy` строит безопасную стратегию из текущих задач и статусов агентов. После этого те же формальные модели продолжают назначать задачи и строить маршруты.

Где смотреть код: `fallback_strategy` и блок `except` в `llm_strategy.py`.

## Возможные упражнения

- Добавить четвертый груз с коротким дедлайном.
- Временно добавить `west_center` в `blocked_edges` и сравнить маршруты с мягким штрафом центра.
- Заменить greedy Contract Net на VCG-механизм.
- Добавить тяжелый груз, который требует коалиции двух роботов.
- Перевести аукцион в полностью распределенный режим, где supervisor только ретранслирует сообщения.
- Заменить локальный `WarehouseRobot` на Webots `youBot`.
- Добавить кеширование стратегий LLM, чтобы не обращаться к модели при неизменном состоянии склада.

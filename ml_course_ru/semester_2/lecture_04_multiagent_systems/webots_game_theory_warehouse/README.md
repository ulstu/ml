# Webots: многоагентная складская доставка с теорией игр и локальной LLM

Этот проект демонстрирует многоагентную транспортную систему для склада в Webots. Несколько мобильных роботов перевозят грузы между зонами склада, договариваются о задачах через сетевые сообщения, выбирают маршруты с помощью формальных моделей теории игр и локально используют LLM как стратегический советник.

Сценарий сделан самодостаточным: роботы описаны локальным `PROTO WarehouseRobot`, поэтому мир не зависит от наличия стандартных PROTO из `/Applications/Webots.app/projects`. Координаты соответствуют Webots ENU: горизонтальная плоскость `XY`, вертикаль `Z`, положительный `X` направлен на восток, положительный `Y` на север.

## Что происходит на сцене

На складе есть четыре робота `ROBOT_0` ... `ROBOT_3`, три груза и несколько доков. Роботы должны выполнить три транспортные задачи:

- `A`: забрать груз в юго-западной зоне склада (`X < 0`, `Y < 0`) и доставить в северо-восточный док (`X > 0`, `Y > 0`).
- `B`: забрать груз в северо-западной зоне (`X < 0`, `Y > 0`) и доставить в юго-восточный док (`X > 0`, `Y < 0`).
- `C`: забрать груз у южного входа центрального коридора (`Y < 0`) и доставить к северному доку (`Y > 0`).

Склад содержит стеллажи, стены, центральный перекресток, объездные коридоры, зоны pickup/dropoff и визуальные маркеры маршрутов. Центральный перекресток намеренно конфликтный: если все роботы выбирают кратчайшие пути, возникают заторы и взаимные уступания.

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
- `traffic_rule` и `note` — текстовая политика и пояснение.

LLM здесь не заменяет формальную модель. Она играет роль адаптивного стратегического слоя: меняет веса и ограничения, а конкретные назначения, маршруты и движение остаются проверяемыми алгоритмами.

Если Ollama недоступна, вернула невалидный JSON или заданный `MAS_LLM_TIMEOUT` истек, используется `fallback_strategy`. Поэтому сценарий остается работоспособным без LLM.

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
    "east_bypass"
  ]
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
safety_margin, traffic_rule, blocked_edges, note.
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
  "blocked_edges": ["west_center"],
  "note": "Prefer bypasses because the center is congested."
}
```

Ответ проходит валидацию:

- веса ограничиваются диапазоном `0.3..4.0`;
- `safety_margin` ограничивается диапазоном `0.45..1.2`;
- `blocked_edges` фильтруется по списку известных ребер;
- если JSON невалиден, используется `fallback_strategy`.

После валидации создается `Strategy`. Именно `Strategy`, а не текст LLM, попадает в формальные модели:

- `deadline_weight` влияет на срочность задач в Contract Net;
- `travel_weight` меняет штраф длины маршрута;
- `congestion_weight` меняет штраф общих коридоров в congestion game;
- `battery_weight` влияет на штраф роботов с низким зарядом;
- `safety_margin` передается в локальные правила движения;
- `blocked_edges` добавляет большой штраф выбранным ребрам.

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
      Сцена склада: стены, стеллажи, доки, грузы, supervisor, четыре робота.

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

По умолчанию контроллер дожидается ответа Ollama:

```bash
export MAS_LLM_TIMEOUT=0
```

Чтобы не блокировать симуляцию слишком долго, можно задать верхний предел:

```bash
export MAS_LLM_TIMEOUT=10
```

Для демонстрации fallback-режима:

```bash
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

- supervisor печатает `[strategy]`, `[dispatch]`, `[assign]`, `[pickup]`, `[done]`, `[success]`;
- LLM-слой печатает `[vector-db]`, `[vector-query]`, `[llm-request]`, `[llm-response]`, `[llm-fallback]`;
- роботы печатают assignment/pickup/done/replan и отправляют статусы.

## Проверка без Webots GUI

Быстрая headless-проверка формальных моделей и логики движения:

```bash
python3 ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/dry_run.py
```

Ожидаемый финал:

```text
SUCCESS completed=A,B,C
```

## Логи LLM и векторной СУБД

Обычный стратегический цикл выглядит так:

```text
[vector-db] path=... db=.../warehouse_rules.sqlite backend=sqlite+pytorch encoder=torch_hashing_bow_v1 documents=6 dimension=192 ids=...
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

- `DEF ROBOT_0` ... `DEF ROBOT_3`;
- поля `translation` и `rotation`;
- сетевые устройства `Receiver` и `Emitter`;
- индивидуальный controller `warehouse_robot_agent`;
- совместимость имен грузов `CARGO_A`, `CARGO_B`, `CARGO_C`.

## Демонстрационные сценарии

Этот раздел можно использовать как план живого показа на занятии. Лучше открыть одновременно Webots-сцену и консоль контроллеров.

### 1. Contract Net Protocol назначает задачи агентам

Что сделать:

```bash
python3 ml_course_ru/semester_2/lecture_04_multiagent_systems/webots_game_theory_warehouse/controllers/warehouse_supervisor/dry_run.py
```

Что показать в логах:

```text
t=000.0 assign R2 -> A via A_south_bypass
t=000.0 assign R3 -> B via B_center
t=000.0 assign R0 -> C via C_center
```

Как объяснить: supervisor собирает `AgentSnapshot`, берет открытые `Task` и запускает `contract_net_auction`. Каждый свободный робот получает оценку `bid = reward + urgency - pickup_cost - battery_penalty`; задача назначается агенту с лучшей ставкой.

Где смотреть код: `controllers/warehouse_supervisor/game_models.py`, функция `contract_net_auction`.

### 2. Congestion game меняет выбор маршрута

Что сделать: запустить Webots и посмотреть строки `[assign]` в консоли supervisor. У каждой задачи есть несколько кандидатных маршрутов: центральный и объездные.

Что показать в логах:

```text
[assign] agent=R2 task=A route=A_south_bypass
[assign] agent=R3 task=B route=B_center
[assign] agent=R0 task=C route=C_center
```

Как объяснить: после выбора победителей аукциона система не просто берет кратчайший путь. `choose_route_nash_equilibrium` несколько раз пересчитывает лучшие ответы агентов. Если общий коридор перегружен, растет `congestion * congestion_weight`, и одному из роботов становится выгоднее взять объезд.

Мини-эксперимент: в `controllers/warehouse_supervisor/llm_strategy.py` или через LLM/fallback поднять `congestion_weight` и сравнить выбранные `route_id`. Чем выше вес затора, тем чаще появляются маршруты с `bypass`.

Где смотреть код: `route_cost` и `choose_route_nash_equilibrium` в `game_models.py`.

### 3. LLM влияет только на стратегические веса

Что сделать: запустить Webots с Ollama и посмотреть стратегический цикл.

Что показать в логах:

```text
[llm-request] model=qwen2.5:0.5b url=http://127.0.0.1:11434/api/generate timeout=wait ...
[llm-response] {"deadline_weight": 1.0, "travel_weight": 2.0, "congestion_weight": 3.0, ...}
[strategy] {"deadline_weight": 1.0, "travel_weight": 2.0, "congestion_weight": 3.0, ...}
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
[vector-db] path=... warehouse_rules.json db=... warehouse_rules.sqlite backend=sqlite+pytorch documents=6 dimension=192 ids=...
[vector-query] query="..." top=collision_dynamic:0.42, traffic_right_hand:0.31, ...
```

Как объяснить: `vector_memory.py` превращает текст правил и текущее состояние склада в PyTorch-векторы, сохраняет embedding в SQLite как BLOB и выбирает top-k правил по cosine similarity. Это локальная векторная СУБД без внешнего сервера: не Chroma/FAISS, но с теми же ключевыми этапами RAG для учебного примера.

Где смотреть код: `LocalVectorStore.query` и `rules_to_prompt` в `vector_memory.py`.

### 5. Supervisor координирует сеть, но не водит роботов

Что сделать: открыть `warehouse_supervisor.py` и `warehouse_robot_agent.py` рядом с Webots.

Что показать в логах:

```text
[strategy] ...
[assign] agent=R2 task=A route=A_south_bypass
[ack] agent=R2 task=A route=A_south_bypass
[pickup] task=A by=R2
[done] task=A by=R2 route=A_south_bypass
```

Как объяснить: supervisor отправляет сообщения `task_board`, `strategy`, `assignment`, но не выполняет покадровое движение робота. Движение, перенос груза и объезд находятся в `warehouse_robot_agent.py`. Это видно по тому, что события `pickup`, `done`, `replan` приходят обратно от роботов.

Где смотреть код: `WarehouseSupervisor.apply_assignment` только отправляет JSON-сообщение, а `WarehouseRobotAgent.step_motion` обновляет состояние конкретного робота.

### 6. Робот локально избегает препятствий и других роботов

Что сделать: во время Webots-запуска смотреть на центральный перекресток и консоль роботов.

Что показать в логах:

```text
[replan] agent=R0 target=[...] detour=...
```

Как объяснить: каждый робот получает статусы соседей, строит `DynamicObstacle`, проверяет следующий сегмент через `segment_is_free`, применяет `TrafficRuleEngine.must_yield`, а при необходимости перестраивает путь через A*.

Если хочется сделать эффект заметнее, можно временно уменьшить свободное пространство в мире или добавить роботу конфликтный маршрут через центральный коридор. Тогда чаще появляются остановки и `replan`.

Где смотреть код: `ensure_collision_free_target`, `peer_obstacles`, `step_motion` в `warehouse_robot_agent.py`; `WarehouseLayout.plan_path` в `planner.py`.

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
- Заставить LLM закрывать ребро `west_center` и сравнить маршруты.
- Заменить greedy Contract Net на VCG-механизм.
- Добавить тяжелый груз, который требует коалиции двух роботов.
- Перевести аукцион в полностью распределенный режим, где supervisor только ретранслирует сообщения.
- Заменить локальный `WarehouseRobot` на Webots `youBot`.
- Добавить кеширование стратегий LLM, чтобы не обращаться к модели при неизменном состоянии склада.

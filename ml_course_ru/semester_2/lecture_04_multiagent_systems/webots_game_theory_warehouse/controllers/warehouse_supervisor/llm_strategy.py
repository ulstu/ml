"""Local LLM strategic advisor for the warehouse supervisor.

The code talks to Ollama on localhost.  If Ollama is not installed, not running,
or the requested model has not been pulled yet, the simulator keeps working with
the deterministic fallback strategy.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Iterable, Optional, Sequence

from game_models import AgentSnapshot, Strategy, Task, distance
from vector_memory import LocalVectorStore, rules_to_prompt


# URL локального Ollama API; можно переопределить через переменную окружения.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
# Модель по умолчанию маленькая, чтобы комфортно работать на Mac M1.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
# MAS_LLM_TIMEOUT управляет ожиданием Ollama: 0/none/wait означает ждать ответ без короткого таймаута.
LLM_TIMEOUT_ENV = os.environ.get("MAS_LLM_TIMEOUT", "0")
# MAS_LLM_DEBUG=1 включает полный prompt и сырой ответ модели.
LLM_DEBUG = os.environ.get("MAS_LLM_DEBUG", "0") == "1"
# MAS_VECTOR_DEBUG=0 отключает подробные логи векторной БД.
VECTOR_DEBUG = os.environ.get("MAS_VECTOR_DEBUG", "1") != "0"


def _log(tag: str, message: str) -> None:
    # Единый формат делает консольные события легко фильтруемыми по [tag].
    print("[%s] %s" % (tag, message))


def _parse_timeout(value: str) -> Optional[float]:
    # Пустые и "ожидающие" значения переводим в None, то есть без короткого timeout.
    normalized = value.strip().lower()
    # Такие значения удобны в shell и читаются как явный режим ожидания.
    if normalized in ("", "0", "none", "wait", "blocking", "inf", "infinite"):
        return None
    try:
        # Остальные значения пытаемся разобрать как число секунд.
        number = float(normalized)
    except ValueError:
        # Ошибка в переменной окружения не должна ломать симуляцию.
        return None
    # Неположительные числа трактуем так же, как режим ожидания без лимита.
    if number <= 0.0:
        return None
    # Положительное число станет верхней границей ожидания Ollama.
    return number


def _timeout_to_log(timeout: Optional[float]) -> str:
    # В логах явно показываем, что запрос не будет прерван через 1 секунду.
    if timeout is None:
        return "wait"
    # Для числового таймаута оставляем компактное представление в секундах.
    return "%.1f" % timeout


# Таймаут вычисляется один раз при старте контроллера из переменной окружения.
LLM_TIMEOUT = _parse_timeout(LLM_TIMEOUT_ENV)


def fallback_strategy(
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
    now: float,
    scene_profile: Optional[dict] = None,
) -> Strategy:
    # open_tasks — задачи, которые еще можно назначить.
    open_tasks = [task for task in tasks if task.status == "open"]
    # late_tasks — открытые задачи с близким дедлайном.
    late_tasks = [task for task in open_tasks if task.deadline - now < 25.0]
    # busy_agents — роботы, которые сейчас не idle.
    busy_agents = [agent for agent in agents if agent.status != "idle"]

    # Чем больше срочных задач, тем выше вес дедлайна.
    deadline_weight = 1.2 + 0.35 * len(late_tasks)
    # Чем больше занятых агентов, тем выше штраф заторов.
    congestion_weight = 1.4 + 0.25 * len(busy_agents)
    # note объясняет студентам, почему fallback выбрал такие веса.
    note = "fallback: late tasks=%d, busy agents=%d" % (len(late_tasks), len(busy_agents))

    # Собираем Strategy в том же формате, что и LLM.
    strategy = Strategy(
        deadline_weight=min(deadline_weight, 3.5),
        travel_weight=1.0,
        congestion_weight=min(congestion_weight, 3.0),
        battery_weight=0.7,
        safety_margin=0.62,
        traffic_rule="deadline_priority_right_hand",
        blocked_edges=(),
        go_agents=(),
        hold_agents=(),
        max_active_tasks=3,
        max_new_assignments=3,
        retrieved_rules=(),
        note=note,
    )
    # Для большой maintenance-сцены fallback тоже должен быть scene-aware:
    # иначе первые назначения могут уйти в центральный конфликт до ответа LLM.
    if scene_profile and scene_profile.get("disrupted_edges"):
        strategy.congestion_weight = max(strategy.congestion_weight, 3.8)
        strategy.travel_weight = min(strategy.travel_weight, 0.55)
        strategy.safety_margin = min(strategy.safety_margin, 0.72)
        strategy.max_active_tasks = 3
        strategy.max_new_assignments = 3
        go_agents, hold_agents, _ = _dispatch_gate(
            tasks,
            agents,
            strategy.max_active_tasks,
            strategy.max_new_assignments,
        )
        strategy.go_agents = go_agents
        strategy.hold_agents = hold_agents
        strategy.note = strategy.note + " | maintenance fallback"
    # Возвращаем нормализованную fallback-стратегию.
    return strategy


def _compact_state(
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
    now: float,
    scene_profile: Optional[dict] = None,
) -> dict:
    # Компактное состояние уменьшает prompt и заставляет LLM отвечать по делу.
    state = {
        # Время округляем до 0.1 секунды.
        "time": round(now, 1),
        # Для каждой задачи оставляем только поля, важные для стратегии.
        "tasks": [
            {
                "id": task.task_id,
                "status": task.status,
                "deadline_in": round(task.deadline - now, 1),
                "reward": task.reward,
                "assigned_agent": task.assigned_agent,
                "route": task.selected_route,
            }
            for task in tasks
        ],
        # Для каждого робота оставляем статус, батарею и текущую задачу.
        "agents": [
            {
                "id": agent.agent_id,
                "status": agent.status,
                "battery": round(agent.battery, 2),
                "task": agent.task_id,
            }
            for agent in agents
        ],
        # possible_edges ограничивает список ребер, которые LLM может закрывать.
        "possible_edges": [
            "west_center",
            "east_center",
            "south_center",
            "north_center",
            "south_bypass",
            "north_bypass",
            "east_bypass",
            "west_bypass",
        ],
        # dispatch_context делает для RAG явным, что нужны правила go/hold-шлюза.
        "dispatch_context": {
            "open_orders": sum(1 for task in tasks if task.status == "open"),
            "idle_agents": sum(1 for agent in agents if agent.status == "idle"),
            "policy_need": "choose go_agents hold_agents max_active_tasks max_new_assignments for pickup dropoff aisle capacity",
        },
    }
    if scene_profile:
        # scene_profile передает LLM именно стратегический контекст эксперимента.
        state["scene_profile"] = scene_profile
    return state


def _extract_json(text: str) -> dict:
    # Иногда LLM добавляет текст вокруг JSON; берем подстроку между первой { и последней }.
    start = text.find("{")
    end = text.rfind("}")
    # Если JSON-объект не найден, считаем ответ невалидным.
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain a JSON object")
    # Парсим извлеченный JSON.
    return json.loads(text[start : end + 1])


def _bounded_weight(value: object, fallback: float) -> float:
    try:
        # Пробуем привести значение LLM к float.
        number = float(value)
    except (TypeError, ValueError):
        # Невалидное значение заменяем fallback-весом.
        return fallback
    # Ограничиваем диапазон, чтобы LLM не сломала формальную модель.
    return max(0.3, min(4.0, number))


def _bounded_margin(value: object, fallback: float) -> float:
    try:
        # safety_margin тоже приходит из JSON и требует проверки типа.
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    # Ограничиваем safety margin реалистичным диапазоном метров.
    return max(0.45, min(0.9, number))


def _bounded_int(value: object, fallback: int, lower: int, upper: int) -> int:
    try:
        # LLM может вернуть число как строку, поэтому приводим через int(float()).
        number = int(float(value))
    except (TypeError, ValueError):
        return fallback
    # Ограничиваем диапазон тактических лимитов.
    return max(lower, min(upper, number))


def _agent_ids(value: object, allowed_agents: set[str]) -> tuple[str, ...]:
    # Невалидные списки заменяем пустым tuple.
    if not isinstance(value, list):
        return ()
    # Оставляем только реально существующих агентов.
    return tuple(str(agent_id) for agent_id in value if str(agent_id) in allowed_agents)


def _task_family(task_id: str) -> str:
    # A2/B3/C4 относятся к тем же pickup/dropoff-семействам, что A/B/C.
    return task_id.rstrip("0123456789")


def _dispatch_gate(
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
    max_active_tasks: int,
    max_new_assignments: int,
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    # active_count показывает, сколько роботов уже едут или везут груз.
    active_count = sum(1 for task in tasks if task.status not in ("open", "completed"))
    # capacity_left — сколько новых роботов можно выпустить сейчас.
    capacity_left = max(max_active_tasks - active_count, 0)
    # allowed_new — итоговый размер новой пачки назначений.
    allowed_new = max(0, min(max_new_assignments, capacity_left))
    # Свободные роботы доступны для нового контракта.
    idle_agents = [agent for agent in agents if agent.status == "idle"]
    # Открытые задачи сортируются так, чтобы сначала покрыть разные зоны A/B/C.
    open_tasks = sorted(
        (task for task in tasks if task.status == "open"),
        key=lambda task: (
            0 if task.task_id == _task_family(task.task_id) else 1,
            task.deadline,
            -task.reward,
            task.task_id,
        ),
    )
    # chosen хранит роботов, которых стоит выпустить прямо сейчас.
    chosen: list[str] = []
    # selected_families не дает первым батчем перегрузить одну pickup/dropoff-пару.
    selected_families: set[str] = set()
    # Для каждой срочной открытой задачи берем ближайшего еще не выбранного робота.
    for task in open_tasks:
        # Останавливаемся, когда окно пропускной способности заполнено.
        if len(chosen) >= allowed_new:
            break
        # В первом проходе пропускаем повтор той же транспортной семьи.
        if _task_family(task.task_id) in selected_families:
            continue
        # candidates — свободные роботы, которые еще не получили разрешение ехать.
        candidates = [agent for agent in idle_agents if agent.agent_id not in chosen]
        # Если свободных роботов больше нет, назначать некого.
        if not candidates:
            break
        # Локальная эвристика выбирает робота с минимальным расстоянием до pickup.
        best_agent = min(
            candidates,
            key=lambda agent: (distance(agent.position, task.pickup), agent.agent_id),
        )
        # Этот робот попадает в go_agents.
        chosen.append(best_agent.agent_id)
        # Запоминаем, что это направление уже покрыто текущим батчем.
        selected_families.add(_task_family(task.task_id))
    # Если семей меньше, чем разрешенных назначений, добираем оставшуюся емкость.
    for task in open_tasks:
        # Останавливаемся, когда окно пропускной способности заполнено.
        if len(chosen) >= allowed_new:
            break
        # candidates — оставшиеся свободные роботы.
        candidates = [agent for agent in idle_agents if agent.agent_id not in chosen]
        # Если свободных роботов больше нет, назначать некого.
        if not candidates:
            break
        # Добор снова идет к ближайшему pickup, но уже без ограничения по семье.
        best_agent = min(
            candidates,
            key=lambda agent: (distance(agent.position, task.pickup), agent.agent_id),
        )
        # Добавляем робота, если он еще не был выбран.
        if best_agent.agent_id not in chosen:
            chosen.append(best_agent.agent_id)
    # go_agents — роботы, которых стратегический слой выпускает в аукцион.
    go_agents = tuple(chosen)
    # Остальные свободные роботы ждут освобождения pickup/dropoff/коридора.
    hold_agents = tuple(agent.agent_id for agent in idle_agents if agent.agent_id not in go_agents)
    return go_agents, hold_agents, allowed_new


def ask_llm_strategy(
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
    now: float,
    timeout: Optional[float] = None,
    scene_profile: Optional[dict] = None,
    policy_guardrails: bool = True,
    rag_enabled: bool = True,
) -> Strategy:
    """Request strategic weights from a local Ollama model."""

    # Если вызывающий код не задал лимит явно, используем общий режим из MAS_LLM_TIMEOUT.
    effective_timeout = LLM_TIMEOUT if timeout is None else timeout
    # Сначала строим компактное состояние склада.
    state = _compact_state(tasks, agents, now, scene_profile)
    # query_text — текстовый запрос для RAG, построенный из состояния склада.
    query_text = json.dumps(state, ensure_ascii=False)
    if rag_enabled:
        # Загружаем локальную векторную БД правил.
        store = LocalVectorStore.default()
        # Достаем top-4 релевантных правила.
        retrieved = store.query(query_text, top_k=4)
        # По умолчанию печатаем статус базы и результаты поиска.
        if VECTOR_DEBUG:
            store.log_status()
            store.log_query(query_text, retrieved)
    else:
        # Ablation: LLM видит состояние и JSON-схему, но не получает документы RAG.
        retrieved = []
    # Prompt явно запрещает LLM отдавать прямые команды моторам.
    prompt = (
        "You are a strategic dispatcher for a multi-agent warehouse. "
        "Return only compact JSON with keys: deadline_weight, travel_weight, "
        "congestion_weight, battery_weight, safety_margin, traffic_rule, "
        "blocked_edges, go_agents, hold_agents, max_active_tasks, "
        "max_new_assignments, note. No markdown, no nested objects. "
        "Weights must be numbers between 0.3 and 4.0. blocked_edges must be a "
        "list containing only known possible_edges; block at most one edge. "
        "go_agents and hold_agents must be arrays of agent id strings such as "
        "[\"R4\", \"R5\"], never arrays of objects. max_active_tasks and "
        "max_new_assignments must be small integers. "
        "safety_margin must be a number between 0.45 and 0.9 meters. Prefer "
        "high deadline_weight when deadlines are close and high "
        "congestion_weight when many agents or tasks share the same central "
        "crossing. In large warehouses, do not over-penalize travel distance: "
        "when the scene_profile says center conflicts are expected, keep "
        "travel_weight low or moderate and use congestion_weight to spread "
        "traffic across bypass routes. Also meter robot release: choose "
        "go_agents that may receive new tasks now and hold_agents that should "
        "wait for pickup/dropoff/aisle space. Choose go_agents by task urgency, "
        "distance to pickup and zone capacity; do not use release waves. "
        "max_active_tasks should usually be 3 for this scene, because the "
        "three pickup/dropoff families A/B/C have one loading slot each. "
        "Use retrieved warehouse rules as policy "
        "memory, but do not invent direct robot commands.\nRetrieved rules:\n"
        + rules_to_prompt(retrieved)
        + "\nState:\n"
        + json.dumps(state, ensure_ascii=True)
    )
    # Payload соответствует Ollama /api/generate.
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 320},
        }
    ).encode("utf-8")
    # Логируем метаданные запроса без полного prompt.
    _log(
        "llm-request",
        "model=%s url=%s timeout=%s tasks=%d agents=%d retrieved=%s"
        % (
            OLLAMA_MODEL,
            OLLAMA_URL,
            _timeout_to_log(effective_timeout),
            len(tasks),
            len(agents),
            ",".join(rule.rule_id for rule in retrieved),
        ),
    )
    # Полный prompt включается только для глубокой отладки.
    if LLM_DEBUG:
        _log("llm-prompt", prompt)

    # urllib.request.Request описывает HTTP POST в Ollama.
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # По умолчанию ждем ответ локальной LLM; числовой timeout можно вернуть через MAS_LLM_TIMEOUT.
        with urllib.request.urlopen(request, timeout=effective_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        # Ollama кладет текстовый ответ модели в поле response.
        raw_response = data.get("response", "{}")
        # Сырой ответ печатаем только в debug-режиме.
        if LLM_DEBUG:
            _log("llm-raw-response", raw_response)
        # Извлекаем JSON-объект из ответа модели.
        decision = _extract_json(raw_response)
        # fallback нужен как источник безопасных значений по умолчанию.
        fallback = fallback_strategy(tasks, agents, now, scene_profile)
        # LLM разрешено закрывать только известные ребра.
        allowed_edges = {
            "west_center",
            "east_center",
            "south_center",
            "north_center",
            "south_bypass",
            "north_bypass",
            "east_bypass",
            "west_bypass",
        }
        # LLM может ссылаться только на реально существующих роботов.
        allowed_agents = {agent.agent_id for agent in agents}
        # Фильтруем blocked_edges, чтобы игнорировать выдуманные LLM ребра.
        blocked = tuple(edge for edge in decision.get("blocked_edges", []) if edge in allowed_edges)[:1]
        # Проверяем списки go/hold агентов.
        go_agents = _agent_ids(decision.get("go_agents"), allowed_agents)
        hold_agents = _agent_ids(decision.get("hold_agents"), allowed_agents)
        # Тактические лимиты пропускной способности.
        max_active_tasks = _bounded_int(decision.get("max_active_tasks"), fallback.max_active_tasks, 1, 4)
        max_new_assignments = _bounded_int(decision.get("max_new_assignments"), fallback.max_new_assignments, 1, 4)
        # Собираем нормализованную Strategy.
        strategy = Strategy(
            deadline_weight=_bounded_weight(decision.get("deadline_weight"), fallback.deadline_weight),
            travel_weight=_bounded_weight(decision.get("travel_weight"), fallback.travel_weight),
            congestion_weight=_bounded_weight(decision.get("congestion_weight"), fallback.congestion_weight),
            battery_weight=_bounded_weight(decision.get("battery_weight"), fallback.battery_weight),
            safety_margin=_bounded_margin(decision.get("safety_margin"), fallback.safety_margin),
            traffic_rule=str(decision.get("traffic_rule", fallback.traffic_rule)),
            blocked_edges=blocked,
            go_agents=go_agents,
            hold_agents=hold_agents,
            max_active_tasks=max_active_tasks,
            max_new_assignments=max_new_assignments,
            retrieved_rules=tuple(rule.rule_id for rule in retrieved),
            note=str(decision.get("note", "ollama strategy")),
        )
        if policy_guardrails and scene_profile and scene_profile.get("disrupted_edges"):
            # Maintenance-профиль основной сцены теперь мягкий: центр получает
            # высокий congestion_weight, но не закрывается полностью, чтобы
            # роботы могли ехать по видимым центральным линиям, если это выгодно.
            strategy.blocked_edges = ()
            strategy.congestion_weight = max(strategy.congestion_weight, 3.8)
            strategy.travel_weight = min(strategy.travel_weight, 0.55)
            strategy.safety_margin = min(strategy.safety_margin, 0.72)
            strategy.max_active_tasks = 3
            strategy.max_new_assignments = 3
            go_agents, hold_agents, _ = _dispatch_gate(
                tasks,
                agents,
                strategy.max_active_tasks,
                strategy.max_new_assignments,
            )
            strategy.go_agents = go_agents
            strategy.hold_agents = hold_agents
            strategy.note = strategy.note + " | maintenance guardrail"
        elif policy_guardrails and scene_profile is None and len(tasks) <= 3:
            # В базовой Webots-демонстрации два диагональных задания иначе
            # целятся в один центральный коридор. Закрываем одно входное ребро
            # как "metering" policy: это не команда роботу, а мягкое
            # стратегическое ограничение для congestion game.
            strategy.blocked_edges = ("west_center",)
            strategy.safety_margin = min(strategy.safety_margin, 0.72)
            go_agents, hold_agents, _ = _dispatch_gate(tasks, agents, 3, 3)
            strategy.go_agents = go_agents
            strategy.hold_agents = hold_agents
            strategy.max_active_tasks = 3
            strategy.max_new_assignments = 3
            strategy.note = strategy.note + " | base-scene guardrail: meter west_center"
        # Логируем уже проверенную стратегию, а не сырой ответ LLM.
        _log("llm-response", strategy_to_log(strategy))
        return strategy
    except (
        urllib.error.URLError,
        TimeoutError,
        socket.timeout,
        ValueError,
        json.JSONDecodeError,
        KeyError,
    ) as error:
        # Любая ошибка LLM переводит систему на deterministic fallback.
        _log("llm-fallback", "reason=%s: %s" % (error.__class__.__name__, error))
        # Строим fallback-стратегию из текущего состояния.
        strategy = fallback_strategy(tasks, agents, now, scene_profile)
        # Даже без LLM сохраняем тот же тактический шлюз пропускной способности.
        go_agents, hold_agents, _ = _dispatch_gate(
            tasks,
            agents,
            strategy.max_active_tasks,
            strategy.max_new_assignments,
        )
        strategy.go_agents = go_agents
        strategy.hold_agents = hold_agents
        # Даже при fallback сохраняем, какие правила были извлечены RAG.
        strategy.retrieved_rules = tuple(rule.rule_id for rule in retrieved)
        # Логируем итоговую fallback-стратегию.
        _log("llm-response", strategy_to_log(strategy))
        return strategy


def strategy_to_log(strategy: Strategy) -> str:
    # asdict превращает dataclass Strategy в обычный dict.
    data = asdict(strategy)
    # tuple не всегда удобно читать в JSON-логе, поэтому делаем list.
    data["blocked_edges"] = list(strategy.blocked_edges)
    # go_agents/hold_agents тоже выводятся списками для наглядной демонстрации.
    data["go_agents"] = list(strategy.go_agents)
    data["hold_agents"] = list(strategy.hold_agents)
    # retrieved_rules показывает, какие документы RAG попали в prompt.
    data["retrieved_rules"] = list(strategy.retrieved_rules)
    # sort_keys делает логи стабильными между запусками.
    return json.dumps(data, ensure_ascii=True, sort_keys=True)

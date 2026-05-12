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

from game_models import AgentSnapshot, Strategy, Task
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


def fallback_strategy(tasks: Sequence[Task], agents: Sequence[AgentSnapshot], now: float) -> Strategy:
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

    # Возвращаем Strategy в том же формате, что и LLM.
    return Strategy(
        deadline_weight=min(deadline_weight, 3.5),
        travel_weight=1.0,
        congestion_weight=min(congestion_weight, 3.0),
        battery_weight=0.7,
        safety_margin=0.62,
        traffic_rule="deadline_priority_right_hand",
        blocked_edges=(),
        retrieved_rules=(),
        note=note,
    )


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


def ask_llm_strategy(
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
    now: float,
    timeout: Optional[float] = None,
    scene_profile: Optional[dict] = None,
) -> Strategy:
    """Request strategic weights from a local Ollama model."""

    # Если вызывающий код не задал лимит явно, используем общий режим из MAS_LLM_TIMEOUT.
    effective_timeout = LLM_TIMEOUT if timeout is None else timeout
    # Сначала строим компактное состояние склада.
    state = _compact_state(tasks, agents, now, scene_profile)
    # Загружаем локальную векторную БД правил.
    store = LocalVectorStore.default()
    # query_text — текстовый запрос для RAG, построенный из состояния склада.
    query_text = json.dumps(state, ensure_ascii=False)
    # Достаем top-4 релевантных правила.
    retrieved = store.query(query_text, top_k=4)
    # По умолчанию печатаем статус базы и результаты поиска.
    if VECTOR_DEBUG:
        store.log_status()
        store.log_query(query_text, retrieved)
    # Prompt явно запрещает LLM отдавать прямые команды моторам.
    prompt = (
        "You are a strategic dispatcher for a multi-agent warehouse. "
        "Return only compact JSON with keys: deadline_weight, travel_weight, "
        "congestion_weight, battery_weight, safety_margin, traffic_rule, "
        "blocked_edges, note. "
        "Weights must be numbers between 0.3 and 4.0. blocked_edges must be a "
        "list containing only known possible_edges; block at most one edge. "
        "safety_margin must be a number between 0.45 and 0.9 meters. Prefer "
        "high deadline_weight when deadlines are close and high "
        "congestion_weight when many agents or tasks share the same central "
        "crossing. In large warehouses, do not over-penalize travel distance: "
        "when the scene_profile says center conflicts are expected, keep "
        "travel_weight low or moderate and use congestion_weight to spread "
        "traffic across bypass routes. Use retrieved warehouse rules as policy "
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
            "options": {"temperature": 0.1, "num_predict": 180},
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
        fallback = fallback_strategy(tasks, agents, now)
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
        # Фильтруем blocked_edges, чтобы игнорировать выдуманные LLM ребра.
        blocked = tuple(edge for edge in decision.get("blocked_edges", []) if edge in allowed_edges)[:1]
        # Собираем нормализованную Strategy.
        strategy = Strategy(
            deadline_weight=_bounded_weight(decision.get("deadline_weight"), fallback.deadline_weight),
            travel_weight=_bounded_weight(decision.get("travel_weight"), fallback.travel_weight),
            congestion_weight=_bounded_weight(decision.get("congestion_weight"), fallback.congestion_weight),
            battery_weight=_bounded_weight(decision.get("battery_weight"), fallback.battery_weight),
            safety_margin=_bounded_margin(decision.get("safety_margin"), fallback.safety_margin),
            traffic_rule=str(decision.get("traffic_rule", fallback.traffic_rule)),
            blocked_edges=blocked,
            retrieved_rules=tuple(rule.rule_id for rule in retrieved),
            note=str(decision.get("note", "ollama strategy")),
        )
        if scene_profile and scene_profile.get("disrupted_edges"):
            # Maintenance-профиль из основной сцены: LLM интерпретирует правило,
            # а guardrail переводит его в безопасное ограничение для игры с
            # заторами. Закрываем только одно ребро, чтобы не обвалить граф.
            disrupted = [edge for edge in scene_profile.get("disrupted_edges", []) if edge in allowed_edges]
            strategy.blocked_edges = tuple(disrupted[:1])
            strategy.congestion_weight = max(strategy.congestion_weight, 3.8)
            strategy.travel_weight = min(strategy.travel_weight, 0.55)
            strategy.safety_margin = min(strategy.safety_margin, 0.72)
            strategy.note = strategy.note + " | maintenance guardrail"
        elif scene_profile is None and len(tasks) <= 3:
            # В базовой Webots-демонстрации два диагональных задания иначе
            # целятся в один центральный коридор. Закрываем одно входное ребро
            # как "metering" policy: это не команда роботу, а мягкое
            # стратегическое ограничение для congestion game.
            strategy.blocked_edges = ("west_center",)
            strategy.safety_margin = min(strategy.safety_margin, 0.72)
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
        strategy = fallback_strategy(tasks, agents, now)
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
    # sort_keys делает логи стабильными между запусками.
    return json.dumps(data, ensure_ascii=True, sort_keys=True)

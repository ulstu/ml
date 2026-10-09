"""Operational traffic rules for multi-robot warehouse motion."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

from game_models import Point, distance


def _deadline_for(task_id: Optional[str], task_deadline: Callable[[str], float]) -> float:
    # Если робот без задачи, он имеет минимальный приоритет.
    if task_id is None:
        return 999.0
    # Иначе получаем дедлайн через callback, который знает task board.
    return task_deadline(task_id)


def _unit_vector(start: Point, target: Point) -> Point:
    # dx — направление по оси X.
    dx = target[0] - start[0]
    # dy — направление по оси Y.
    dy = target[1] - start[1]
    # length защищен от нуля, чтобы не делить на 0.
    length = max(math.hypot(dx, dy), 1e-9)
    # Возвращаем нормированный вектор направления.
    return (dx / length, dy / length)


def _has_other_on_right(my_heading: Point, relative_other: Point) -> bool:
    # 2D cross product показывает, с какой стороны находится другой робот.
    cross = my_heading[0] * relative_other[1] - my_heading[1] * relative_other[0]
    # При нашей ENU-ориентации отрицательный cross означает "справа".
    return cross < 0.0


class TrafficRuleEngine:
    def __init__(self, safety_margin: float = 0.62) -> None:
        # safety_margin задает базовую дистанцию, при которой включаются правила.
        self.safety_margin = safety_margin

    def must_yield(
        self,
        agent: object,
        target: Point,
        agents: Iterable[object],
        task_deadline: Callable[[str], float],
    ) -> bool:
        # my_heading — куда собирается двигаться текущий робот.
        my_heading = _unit_vector(agent.position, target)
        # my_deadline — дедлайн задачи текущего робота.
        my_deadline = _deadline_for(agent.task_id, task_deadline)
        # Загруженный робот получает приоритет при равном дедлайне.
        my_loaded = agent.status == "transporting"

        # Сравниваем текущего робота с каждым другим участником движения.
        for other in agents:
            # Самого себя и idle-роботов игнорируем.
            if other.agent_id == agent.agent_id or other.status == "idle":
                continue
            # separation — текущее расстояние между роботами.
            separation = distance(agent.position, other.position)
            # Если робот далеко, он не влияет на локальное правило.
            if separation > self.safety_margin * 2.4:
                continue

            # Дедлайн другого робота нужен для приоритета срочности.
            other_deadline = _deadline_for(other.task_id, task_deadline)
            # Загруженность другого робота нужна для приоритета груза.
            other_loaded = other.status == "transporting"
            # relative_other — вектор от текущего робота к другому.
            relative_other = (
                other.position[0] - agent.position[0],
                other.position[1] - agent.position[1],
            )

            # Пустой робот уступает загруженному: это быстрее освобождает dropoff-зону.
            if other_loaded and not my_loaded:
                return True
            # Загруженный робот не уступает пустому только из-за более позднего дедлайна.
            if my_loaded and not other_loaded:
                continue
            # Менее срочный робот уступает более срочному, когда статус загрузки одинаков.
            if my_deadline > other_deadline + 1e-6:
                return True
            # При равных условиях применяем правило правой стороны.
            if (
                abs(my_deadline - other_deadline) <= 1e-6
                and other_loaded == my_loaded
                and _has_other_on_right(my_heading, relative_other)
            ):
                return True
            # Последний tie-breaker делает поведение детерминированным.
            if (
                abs(my_deadline - other_deadline) <= 1e-6
                and other_loaded == my_loaded
                and agent.agent_id > other.agent_id
            ):
                return True
        # Если ни одно правило уступания не сработало, робот может ехать.
        return False

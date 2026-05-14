"""Shared warehouse scenario data for Webots and dry-run tests."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from game_models import Route, Task
from warehouse_geometry import SCENE_SCALE, scaled_deadline, scaled_point


WEBOTS_TASK_REPEATS = 3
WEBOTS_ROBOT_COUNT = 8
WEBOTS_ROUTE_VARIANT = "redundant"
WEBOTS_DISRUPTED_EDGES = ("west_center", "east_center")
WEBOTS_SCENE_PROFILE = {
    "name": "large_maintenance_9_tasks_8_robots",
    "scale": SCENE_SCALE,
    "floor_area_multiplier": round(SCENE_SCALE * SCENE_SCALE, 2),
    "robots": WEBOTS_ROBOT_COUNT,
    "tasks": WEBOTS_TASK_REPEATS * 3,
    "route_variant": WEBOTS_ROUTE_VARIANT,
    "layout_variant": "maintenance_graph",
    "center_conflicts_expected": True,
    "disrupted_edges": list(WEBOTS_DISRUPTED_EDGES),
    "hypothesis": "LLM+RAG should shift flow from delayed center edges and meter robot release to pickup/dropoff zones.",
}


def _p(x: float, y: float, scale: float) -> tuple[float, float]:
    # Helper keeps the route tables readable in compact-scene coordinates.
    return scaled_point((x, y), scale)


def make_tasks(scale: float = SCENE_SCALE) -> List[Task]:
    # Возвращаем новый список задач при каждом запуске, чтобы статусы не переиспользовались.
    return [
        # A: юго-западный pickup -> северо-восточный dropoff.
        Task("A", pickup=_p(-5.36, -2.40, scale), dropoff=_p(5.36, 2.48, scale), reward=12.0, deadline=scaled_deadline(80.0, scale), cargo_def="CARGO_A"),
        # B: северо-западный pickup -> юго-восточный dropoff.
        Task("B", pickup=_p(-5.36, 2.40, scale), dropoff=_p(5.36, -2.48, scale), reward=10.0, deadline=scaled_deadline(95.0, scale), cargo_def="CARGO_B"),
        # C: южный вход центрального коридора -> северный док.
        Task("C", pickup=_p(0.00, -4.24, scale), dropoff=_p(0.00, 4.24, scale), reward=14.0, deadline=scaled_deadline(100.0, scale), cargo_def="CARGO_C"),
    ]


def _offset_task(task: Task, repeat: int) -> Task:
    # Повторные грузы лежат рядом с одной pickup-зоной, но не накладываются.
    offsets = {
        "A": ((0.0, 0.0), (0.0, -1.05), (0.0, 1.05)),
        "B": ((0.0, 0.0), (0.0, 1.05), (0.0, -1.05)),
        "C": ((0.0, 0.0), (-1.05, 0.0), (1.05, 0.0)),
    }
    suffix = "" if repeat == 0 else str(repeat + 1)
    dx, dy = offsets[task.task_id][repeat]
    # Дедлайн сдвигается по копиям, чтобы Contract Net видел разную срочность.
    deadline_shift = scaled_deadline(18.0) * repeat
    # Малое снижение reward помогает сделать ставки разных копий неодинаковыми.
    reward_decay = 0.35 * repeat
    return Task(
        task_id=task.task_id + suffix,
        pickup=(round(task.pickup[0] + dx, 3), round(task.pickup[1] + dy, 3)),
        dropoff=task.dropoff,
        reward=max(1.0, task.reward - reward_decay),
        deadline=task.deadline + deadline_shift,
        cargo_def=task.cargo_def + suffix,
        status="open",
    )


def make_repeated_tasks(repeats: int = WEBOTS_TASK_REPEATS, scale: float = SCENE_SCALE) -> List[Task]:
    # Повторяет A/B/C несколькими копиями для плотной сцены; все задачи открыты сразу.
    tasks: List[Task] = []
    for repeat in range(repeats):
        for task in make_tasks(scale):
            tasks.append(_offset_task(task, repeat))
    return tasks


def make_routes(scale: float = SCENE_SCALE, route_variant: str = "standard") -> Dict[str, Sequence[Route]]:
    # Ключ словаря — task_id, значение — допустимые маршруты для этой задачи.
    routes: Dict[str, Sequence[Route]] = {
        # Для A есть короткий маршрут через центр и южный обход.
        "A": (
            Route(
                # A_center использует общий центральный перекресток.
                "A_center",
                # Waypoints идут по синей центральной разметке: запад -> центр -> восток.
                (_p(-5.36, -2.40, scale), _p(-5.36, 0.00, scale), _p(-0.80, 0.00, scale), _p(0.80, 0.00, scale), _p(5.36, 0.00, scale), _p(5.36, 2.48, scale)),
                # Edges нужны для расчета congestion cost.
                ("west_center", "south_center", "east_center"),
            ),
            Route(
                # A_south_bypass длиннее, но разгружает центр.
                "A_south_bypass",
                (_p(-5.36, -2.40, scale), _p(-3.20, -4.64, scale), _p(4.40, -4.64, scale), _p(4.40, 2.48, scale), _p(5.36, 2.48, scale)),
                ("south_bypass",),
            ),
        ),
        # Для B есть центральный маршрут и северный обход.
        "B": (
            Route(
                # B_center пересекает центральный узел склада.
                "B_center",
                (_p(-5.36, 2.40, scale), _p(-5.36, 0.00, scale), _p(-0.80, 0.00, scale), _p(0.80, 0.00, scale), _p(5.36, 0.00, scale), _p(5.36, -2.48, scale)),
                ("west_center", "north_center", "east_center"),
            ),
            Route(
                # B_north_bypass уводит робота по северному коридору.
                "B_north_bypass",
                (_p(-5.36, 2.40, scale), _p(-3.20, 4.64, scale), _p(5.80, 4.64, scale), _p(5.80, -2.48, scale), _p(5.36, -2.48, scale)),
                ("north_bypass",),
            ),
        ),
        # Для C есть прямой вертикальный маршрут и восточный обход.
        "C": (
            Route(
                # C_center проходит через самый загруженный центр склада.
                "C_center",
                (_p(0.0, -4.24, scale), _p(0.0, -0.80, scale), _p(0.0, 0.80, scale), _p(0.0, 4.24, scale)),
                ("south_center", "north_center"),
            ),
            Route(
                # C_east_bypass обходит центр через восточный коридор.
                "C_east_bypass",
                (_p(0.0, -4.24, scale), _p(4.40, -4.24, scale), _p(4.40, 4.24, scale), _p(0.0, 4.24, scale)),
                ("east_bypass",),
            ),
        ),
    }
    if route_variant in ("redundant", "wide_ring"):
        # Дополнительные маршруты нужны для эксперимента: LLM может выиграть,
        # только если стратегическому уровню есть из чего выбирать.
        routes["A"] = routes["A"] + (
            Route(
                "A_north_ring",
                (_p(-5.36, -2.40, scale), _p(-6.20, -1.20, scale), _p(-6.20, 4.64, scale), _p(5.36, 4.64, scale), _p(5.36, 2.48, scale)),
                ("west_bypass", "north_bypass"),
            ),
        )
        routes["B"] = routes["B"] + (
            Route(
                "B_south_ring",
                (_p(-5.36, 2.40, scale), _p(-6.20, 1.20, scale), _p(-6.20, -4.64, scale), _p(5.36, -4.64, scale), _p(5.36, -2.48, scale)),
                ("west_bypass", "south_bypass"),
            ),
        )
        routes["C"] = routes["C"] + (
            Route(
                "C_west_bypass",
                (_p(0.0, -4.24, scale), _p(-4.40, -4.24, scale), _p(-4.40, 4.24, scale), _p(0.0, 4.24, scale)),
                ("west_bypass",),
            ),
        )
    return routes


def make_repeated_routes(
    repeats: int = WEBOTS_TASK_REPEATS,
    scale: float = SCENE_SCALE,
    route_variant: str = WEBOTS_ROUTE_VARIANT,
) -> Dict[str, Sequence[Route]]:
    # Создает маршруты для задач A/B/C, A2/B2/C2, A3/B3/C3.
    base_routes = make_routes(scale, route_variant)
    repeated_tasks = make_repeated_tasks(repeats, scale)
    routes_by_task: Dict[str, Sequence[Route]] = {}
    for task in repeated_tasks:
        base_task_id = task.task_id[0]
        suffix = task.task_id[1:]
        task_routes: List[Route] = []
        for route in base_routes[base_task_id]:
            route_id = route.route_id if not suffix else "%s_%s" % (route.route_id, suffix)
            # Первая точка маршрута совпадает с конкретным crate pickup этой копии.
            waypoints = (task.pickup,) + tuple(route.waypoints[1:-1]) + (task.dropoff,)
            task_routes.append(Route(route_id, waypoints, route.edges))
        routes_by_task[task.task_id] = tuple(task_routes)
    return routes_by_task


def make_webots_tasks_and_routes() -> Tuple[List[Task], Dict[str, Sequence[Route]]]:
    # Основная Webots-сцена теперь соответствует large_maintenance_9_tasks_8_robots.
    return make_repeated_tasks(WEBOTS_TASK_REPEATS), make_repeated_routes(WEBOTS_TASK_REPEATS)

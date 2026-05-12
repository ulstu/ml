"""Shared warehouse scenario data for Webots and dry-run tests."""

from __future__ import annotations

from typing import Dict, List, Sequence

from game_models import Route, Task


def make_tasks() -> List[Task]:
    # Возвращаем новый список задач при каждом запуске, чтобы статусы не переиспользовались.
    return [
        # A: юго-западный pickup -> северо-восточный dropoff.
        Task("A", pickup=(-5.36, -2.40), dropoff=(5.36, 2.48), reward=12.0, deadline=80.0, cargo_def="CARGO_A"),
        # B: северо-западный pickup -> юго-восточный dropoff.
        Task("B", pickup=(-5.36, 2.40), dropoff=(5.36, -2.48), reward=10.0, deadline=95.0, cargo_def="CARGO_B"),
        # C: южный вход центрального коридора -> северный док.
        Task("C", pickup=(0.00, -4.24), dropoff=(0.00, 4.24), reward=14.0, deadline=100.0, cargo_def="CARGO_C"),
    ]


def make_routes() -> Dict[str, Sequence[Route]]:
    # Ключ словаря — task_id, значение — допустимые маршруты для этой задачи.
    return {
        # Для A есть короткий маршрут через центр и южный обход.
        "A": (
            Route(
                # A_center использует общий центральный перекресток.
                "A_center",
                # Waypoints идут от pickup к dropoff через центральные проходы.
                ((-5.36, -2.40), (-2.00, -2.40), (-2.00, 0.0), (0.0, 0.0), (2.00, 0.0), (2.00, 2.48), (5.36, 2.48)),
                # Edges нужны для расчета congestion cost.
                ("west_center", "south_center", "east_center"),
            ),
            Route(
                # A_south_bypass длиннее, но разгружает центр.
                "A_south_bypass",
                ((-5.36, -2.40), (-3.20, -4.64), (5.36, -4.64), (5.36, 2.48)),
                ("south_bypass",),
            ),
        ),
        # Для B есть центральный маршрут и северный обход.
        "B": (
            Route(
                # B_center пересекает центральный узел склада.
                "B_center",
                ((-5.36, 2.40), (-2.00, 2.40), (-2.00, 0.0), (0.0, 0.0), (2.00, 0.0), (2.00, -2.48), (5.36, -2.48)),
                ("west_center", "north_center", "east_center"),
            ),
            Route(
                # B_north_bypass уводит робота по северному коридору.
                "B_north_bypass",
                ((-5.36, 2.40), (-3.20, 4.64), (5.36, 4.64), (5.36, -2.48)),
                ("north_bypass",),
            ),
        ),
        # Для C есть прямой вертикальный маршрут и восточный обход.
        "C": (
            Route(
                # C_center проходит через самый загруженный центр склада.
                "C_center",
                ((0.0, -4.24), (0.0, -2.00), (0.0, 0.0), (0.0, 2.00), (0.0, 4.24)),
                ("south_center", "north_center"),
            ),
            Route(
                # C_east_bypass обходит центр через восточный коридор.
                "C_east_bypass",
                ((0.0, -4.24), (4.40, -4.24), (4.40, 4.24), (0.0, 4.24)),
                ("east_bypass",),
            ),
        ),
    }

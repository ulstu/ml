"""Grid planner and geometric collision checks for the warehouse example."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from game_models import Point, distance
from warehouse_geometry import SCENE_SCALE, scaled_point


@dataclass(frozen=True)
class RectObstacle:
    # name нужен для отладки карты препятствий.
    name: str
    # center — центр прямоугольника-препятствия на плоскости склада.
    center: Point
    # size — ширина и высота прямоугольника в метрах по X и Y.
    size: Point

    def contains(self, point: Point, margin: float = 0.0) -> bool:
        # margin расширяет препятствие на радиус безопасности робота.
        half_x = self.size[0] / 2.0 + margin
        # Аналогично расширяем препятствие по оси Y.
        half_y = self.size[1] / 2.0 + margin
        # Точка занята, если она попадает внутрь расширенного прямоугольника.
        return (
            self.center[0] - half_x <= point[0] <= self.center[0] + half_x
            and self.center[1] - half_y <= point[1] <= self.center[1] + half_y
        )


@dataclass(frozen=True)
class DynamicObstacle:
    # agent_id помогает понять, какой робот стал динамическим препятствием.
    agent_id: str
    # position — текущая позиция другого робота.
    position: Point
    # radius — радиус safety disk другого робота.
    radius: float


class WarehouseLayout:
    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        obstacles: Sequence[RectObstacle],
        grid_step: float = 0.35,
        robot_radius: float = 0.46,
    ) -> None:
        # bounds задает прямоугольник допустимой области: min_x, max_x, min_y, max_y.
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        # obstacles — статические препятствия: стеллажи и другие непроезжие зоны.
        self.obstacles = list(obstacles)
        # grid_step — размер клетки A* в метрах.
        self.grid_step = grid_step
        # robot_radius — радиус safety disk робота.
        self.robot_radius = robot_radius

    def in_bounds(self, point: Point, margin: float = 0.0) -> bool:
        # Проверяем, остается ли safety disk внутри допустимых границ склада.
        return (
            self.min_x + margin <= point[0] <= self.max_x - margin
            and self.min_y + margin <= point[1] <= self.max_y - margin
        )

    def point_is_free(
        self,
        point: Point,
        margin: Optional[float] = None,
        dynamic: Sequence[DynamicObstacle] = (),
    ) -> bool:
        # Если margin не задан явно, используем физический радиус безопасности робота.
        safety = self.robot_radius if margin is None else margin
        # Сначала отсекаем точки за пределами склада.
        if not self.in_bounds(point, safety):
            return False
        # Затем проверяем пересечение safety disk со статическими препятствиями.
        if any(obstacle.contains(point, safety) for obstacle in self.obstacles):
            return False
        # После этого проверяем динамические препятствия — других роботов.
        for obstacle in dynamic:
            # Если расстояние между центрами меньше суммы радиусов, есть коллизия.
            if distance(point, obstacle.position) < safety + obstacle.radius:
                return False
        # Если ни одна проверка не сработала, точка свободна.
        return True

    def segment_is_free(
        self,
        start: Point,
        goal: Point,
        margin: Optional[float] = None,
        dynamic: Sequence[DynamicObstacle] = (),
    ) -> bool:
        # Длина сегмента нужна, чтобы понять, сколько промежуточных точек проверять.
        length = distance(start, goal)
        # Чем меньше шаг семплирования, тем меньше шанс "перепрыгнуть" угол стеллажа.
        # В большой сцене grid_step крупный, поэтому берем 0.20 клетки: это дороже,
        # но не дает длинному диагональному сегменту ошибочно пройти через safety margin.
        samples = max(2, int(math.ceil(length / (self.grid_step * 0.20))))
        # Проверяем start, goal и все промежуточные точки на отрезке.
        for index in range(samples + 1):
            # alpha идет от 0 до 1 и задает положение точки на отрезке.
            alpha = index / samples
            # Линейная интерполяция между start и goal.
            point = (
                start[0] + (goal[0] - start[0]) * alpha,
                start[1] + (goal[1] - start[1]) * alpha,
            )
            # Если хотя бы одна точка занята, весь сегмент считается небезопасным.
            if not self.point_is_free(point, margin, dynamic):
                return False
        # Все семплы свободны — отрезок можно пройти.
        return True

    def _grid_key(self, point: Point) -> Tuple[int, int]:
        # Переводим метрические координаты в целочисленный индекс клетки.
        return (
            int(round((point[0] - self.min_x) / self.grid_step)),
            int(round((point[1] - self.min_y) / self.grid_step)),
        )

    def _point_from_key(self, key: Tuple[int, int]) -> Point:
        # Переводим индекс клетки обратно в метрические координаты ее центра.
        return (
            self.min_x + key[0] * self.grid_step,
            self.min_y + key[1] * self.grid_step,
        )

    def _nearest_free_key(
        self,
        point: Point,
        dynamic: Sequence[DynamicObstacle],
        search_radius: int = 5,
    ) -> Tuple[int, int]:
        # Базовая клетка — ближайшая к исходной метрической точке.
        base = self._grid_key(point)
        # Пока свободная клетка не найдена, считаем лучшей базовую.
        best_key = base
        # best_distance нужен, чтобы выбрать ближайшую свободную клетку.
        best_distance = float("inf")
        # Расширяем квадрат поиска вокруг базовой клетки.
        for radius in range(search_radius + 1):
            # dx перебирает смещение по X в клетках.
            for dx in range(-radius, radius + 1):
                # dy перебирает смещение по Y в клетках.
                for dy in range(-radius, radius + 1):
                    # key — кандидатная клетка.
                    key = (base[0] + dx, base[1] + dy)
                    # candidate — метрическая точка центра кандидатной клетки.
                    candidate = self._point_from_key(key)
                    # Занятые клетки не подходят как старт/финиш A*.
                    if not self.point_is_free(candidate, dynamic=dynamic):
                        continue
                    # Считаем расстояние от исходной точки до кандидатной клетки.
                    candidate_distance = distance(point, candidate)
                    # Сохраняем ближайшую свободную клетку.
                    if candidate_distance < best_distance:
                        best_key = key
                        best_distance = candidate_distance
            # Как только на текущем радиусе нашли свободную клетку, возвращаем ее.
            if best_distance < float("inf"):
                return best_key
        # Fallback: возвращаем базовую клетку, даже если она неидеальна.
        return best_key

    def plan_path(
        self,
        start: Point,
        goal: Point,
        dynamic: Sequence[DynamicObstacle] = (),
        max_expansions: int = 7000,
    ) -> List[Point]:
        # Если прямой сегмент безопасен, A* не нужен.
        if self.segment_is_free(start, goal, dynamic=dynamic):
            return [goal]

        # A* должен стартовать из свободной клетки рядом с текущей позицией.
        start_key = self._nearest_free_key(start, dynamic)
        # Цель ищем без динамических препятствий, чтобы робот мог ждать освобождения.
        goal_key = self._nearest_free_key(goal, ())
        # frontier — приоритетная очередь A*: (priority, cell_key).
        frontier: List[Tuple[float, Tuple[int, int]]] = [(0.0, start_key)]
        # came_from хранит дерево предков для восстановления пути.
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_key: None}
        # cost_so_far хранит лучшую известную стоимость пути до клетки.
        cost_so_far: Dict[Tuple[int, int], float] = {start_key: 0.0}
        # 8-связная сетка: можно идти по горизонтали, вертикали и диагонали.
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        # expansions ограничивает поиск, чтобы контроллер не завис при плохой карте.
        expansions = 0
        # Основной цикл A* работает, пока есть клетки на фронтире.
        while frontier and expansions < max_expansions:
            # Берем клетку с минимальным приоритетом.
            _, current = heapq.heappop(frontier)
            # Увеличиваем счетчик раскрытых клеток.
            expansions += 1
            # Если дошли до цели, можно завершать поиск.
            if current == goal_key:
                break
            # Перебираем соседей текущей клетки.
            for dx, dy in neighbors:
                # next_key — индекс соседней клетки.
                next_key = (current[0] + dx, current[1] + dy)
                # current_point — метрическая точка текущей клетки.
                current_point = self._point_from_key(current)
                # next_point — метрическая точка соседней клетки.
                next_point = self._point_from_key(next_key)
                # Запрещаем переход, если короткий сегмент пересекает препятствие.
                if not self.segment_is_free(current_point, next_point, dynamic=dynamic):
                    continue
                # Диагональный шаг дороже прямого.
                step_cost = math.hypot(dx, dy) * self.grid_step
                # new_cost — стоимость пути до next_key через current.
                new_cost = cost_so_far[current] + step_cost
                # Обновляем клетку, если она новая или найден путь дешевле.
                if next_key not in cost_so_far or new_cost < cost_so_far[next_key]:
                    cost_so_far[next_key] = new_cost
                    # priority = g + h, где h — евклидова эвристика до цели.
                    priority = new_cost + distance(next_point, goal)
                    # Добавляем соседа во фронтир A*.
                    heapq.heappush(frontier, (priority, next_key))
                    # Запоминаем предка для восстановления пути.
                    came_from[next_key] = current

        # Если A* не нашел путь, возвращаем исходную цель как fallback.
        if goal_key not in came_from:
            return [goal]

        # Восстанавливаем путь от цели к старту по came_from.
        keys: List[Tuple[int, int]] = []
        # current начинает в целевой клетке.
        current: Optional[Tuple[int, int]] = goal_key
        # Идем назад до стартовой клетки.
        while current is not None:
            keys.append(current)
            current = came_from[current]
        # Разворачиваем путь и переводим клетки в метрические точки.
        points = [self._point_from_key(key) for key in reversed(keys)]
        # Первую точку заменяем реальной позицией робота.
        if points:
            points[0] = start
            # Последнюю точку заменяем реальной целью, а не центром клетки.
            points[-1] = goal
        # Упрощаем ломаную, убирая лишние промежуточные точки.
        return self._simplify(points, dynamic)

    def _simplify(self, points: Sequence[Point], dynamic: Sequence[DynamicObstacle]) -> List[Point]:
        # Для короткого пути возвращаем все точки кроме текущей позиции.
        if len(points) <= 2:
            return list(points[1:])
        # simplified будет содержать только необходимые поворотные точки.
        simplified: List[Point] = []
        # anchor_index — точка, от которой пытаемся провести максимально длинный сегмент.
        anchor_index = 0
        # Идем по пути до последней точки.
        while anchor_index < len(points) - 1:
            # Сначала пробуем сразу соединиться с самой дальней точкой.
            next_index = len(points) - 1
            # Если прямой сегмент запрещен, двигаем next_index ближе.
            while next_index > anchor_index + 1:
                if self.segment_is_free(points[anchor_index], points[next_index], dynamic=dynamic):
                    break
                next_index -= 1
            # Добавляем самую дальнюю достижимую точку.
            simplified.append(points[next_index])
            # Новый anchor — добавленная точка.
            anchor_index = next_index
        # Возвращаем упрощенную последовательность waypoints.
        return simplified


def make_warehouse_layout(scale: float = SCENE_SCALE, route_variant: str = "standard") -> WarehouseLayout:
    # Фабрика возвращает карту склада, совпадающую с расширенной Webots-сценой.
    obstacles = [
        RectObstacle("rack_north_west", scaled_point((-3.12, 3.44), scale), scaled_point((3.28, 0.36), scale)),
        RectObstacle("rack_north_east", scaled_point((3.12, 3.44), scale), scaled_point((3.28, 0.36), scale)),
        RectObstacle("rack_south_west", scaled_point((-3.12, -3.44), scale), scaled_point((3.28, 0.36), scale)),
        RectObstacle("rack_south_east", scaled_point((3.12, -3.44), scale), scaled_point((3.28, 0.36), scale)),
    ]
    if route_variant == "maintenance_center":
        obstacles.append(RectObstacle("maintenance_center", scaled_point((0.0, 0.0), scale), scaled_point((1.55, 1.55), scale)))
    return WarehouseLayout(
        # Границы допустимой области движения роботов.
        bounds=(-7.45 * scale, 7.45 * scale, -5.85 * scale, 5.85 * scale),
        # Прямоугольные стеллажи, которые нельзя пересекать safety disk робота.
        obstacles=obstacles,
        # На увеличенной сцене крупная клетка держит A* быстрым и наглядным.
        grid_step=0.75 if scale >= 2.0 else 0.35,
        # Радиус safety disk чуть больше габаритов визуальной платформы.
        robot_radius=0.46,
    )

"""Headless verification for the warehouse MAS scenario.

This script runs the same tasks, routes and game-theoretic allocation model as
the Webots supervisor, but without importing the Webots `controller` module.
It is useful for CI, lectures and quick debugging when the GUI simulator is not
available from the terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from game_models import AgentSnapshot, Point, Strategy, contract_net_auction, distance
from planner import DynamicObstacle, make_warehouse_layout
from scenario import WEBOTS_ROBOT_COUNT, make_webots_tasks_and_routes
from traffic_rules import TrafficRuleEngine
from warehouse_geometry import make_agent_start_positions


# Скорость совпадает с Webots-контроллером робота.
ROBOT_SPEED = 0.48
# Период запуска аукциона совпадает с coordinator.
DISPATCH_PERIOD = 5.0
# Радиус достижения waypoint совпадает с Webots-контроллером.
ARRIVAL_EPS = 0.08


@dataclass
class DryAgent:
    # agent_id — имя робота в headless-модели.
    agent_id: str
    # position — координата робота на плоскости X-Y.
    position: Point
    # battery — упрощенная модель батареи.
    battery: float = 1.0
    # status повторяет состояния Webots-агента.
    status: str = "idle"
    # task_id — текущая задача, если робот занят.
    task_id: Optional[str] = None
    # route_id — текущий маршрут.
    route_id: Optional[str] = None
    # waypoints — локально спланированный маршрут.
    waypoints: List[Point] = field(default_factory=list)
    # waypoint_index — текущая цель внутри маршрута.
    waypoint_index: int = 0
    # wait_timer нужен для диагностики ожиданий.
    wait_timer: float = 0.0


def snapshots(agents: List[DryAgent]) -> List[AgentSnapshot]:
    # Преобразуем dry-run агентов в формат формальной модели.
    return [
        AgentSnapshot(agent.agent_id, agent.position, agent.battery, agent.status, agent.task_id)
        for agent in agents
    ]


def dynamic_obstacles(agent: DryAgent, agents: List[DryAgent], layout) -> List[DynamicObstacle]:
    # Другие активные агенты становятся динамическими препятствиями.
    return [
        DynamicObstacle(other.agent_id, other.position, layout.robot_radius)
        for other in agents
        if other.agent_id != agent.agent_id and other.status != "idle"
    ]


def retreat_from_loaded_peer(agent: DryAgent, agents: List[DryAgent], layout, dt: float) -> bool:
    # Пустой робот может оказаться прямо в зоне разгрузки загруженного робота.
    if agent.status == "transporting":
        return False
    # Ищем ближайшего загруженного робота в локальной safety-зоне.
    loaded_peers = [
        other
        for other in agents
        if other.agent_id != agent.agent_id
        and other.status == "transporting"
        and distance(agent.position, other.position) <= layout.robot_radius * 3.0
    ]
    # Если рядом нет загруженного робота, обычное ожидание корректно.
    if not loaded_peers:
        return False
    # Ближайший загруженный робот определяет направление освобождения.
    peer = min(loaded_peers, key=lambda other: distance(agent.position, other.position))
    # Вектор направлен от загруженного робота к текущему пустому роботу.
    dx = agent.position[0] - peer.position[0]
    dy = agent.position[1] - peer.position[1]
    # Если центры почти совпали, используем устойчивое направление вверх по Y.
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    # Отступление короткое: это не новый маршрут, а освобождение локального конфликта.
    step = ROBOT_SPEED * dt
    # candidate — точка, где пустой робот даст загруженному больше пространства.
    candidate = (agent.position[0] + dx / length * step, agent.position[1] + dy / length * step)
    # Проверяем статическую безопасность точки отступления.
    if not layout.point_is_free(candidate):
        return False
    # Не учитываем самого загруженного робота на отрезке: цель именно увеличить дистанцию от него.
    if not layout.segment_is_free(agent.position, candidate, dynamic=()):
        return False
    # Применяем короткий шаг освобождения зоны.
    agent.position = candidate
    # Батарея расходуется так же, как при обычном движении.
    agent.battery = max(0.05, agent.battery - 0.00035 * step)
    # Сброс wait_timer показывает, что агент активно освободил конфликт.
    agent.wait_timer = 0.0
    return True


def plan_operational_waypoints(layout, start: Point, goals: List[Point]) -> List[Point]:
    # cursor хранит начало следующего сегмента.
    cursor = start
    # planned собирает итоговую последовательность waypoints.
    planned: List[Point] = []
    # Каждый стратегический waypoint уточняется A*.
    for goal in goals:
        planned.extend(layout.plan_path(cursor, goal))
        cursor = goal
    # Возвращаем локально безопасный маршрут.
    return planned


def move_towards(agent: DryAgent, agents: List[DryAgent], layout, traffic: TrafficRuleEngine, task_deadline, dt: float) -> None:
    # Если маршрута больше нет, двигаться некуда.
    if agent.waypoint_index >= len(agent.waypoints):
        return
    # target — текущий waypoint.
    target = agent.waypoints[agent.waypoint_index]
    # При достижении waypoint переходим к следующему.
    if distance(agent.position, target) < ARRIVAL_EPS:
        agent.position = target
        agent.waypoint_index += 1
        return

    # Правила движения могут заставить агента ждать.
    if traffic.must_yield(agent, target, agents, task_deadline):
        # Если пустой робот блокирует загруженного, он должен освободить место, а не стоять.
        if retreat_from_loaded_peer(agent, agents, layout, dt):
            return
        agent.wait_timer += dt
        return

    # Строим список динамических препятствий из других агентов.
    dynamic = dynamic_obstacles(agent, agents, layout)
    # Длинный сегмент проверяем только на статические препятствия. Динамику
    # учитываем на коротком следующем шаге, чтобы не перестраиваться из-за
    # робота, который пока далеко, но стоит около будущего waypoint.
    if not layout.segment_is_free(agent.position, target, dynamic=()):
        detour = layout.plan_path(agent.position, target, dynamic=())
        # Если detour пуст или первый шаг небезопасен, агент ждет.
        if not detour or not layout.segment_is_free(agent.position, detour[0], dynamic=dynamic):
            agent.wait_timer += dt
            return
        # Подменяем текущий хвост маршрута новым объездом.
        agent.waypoints = detour + agent.waypoints[agent.waypoint_index + 1 :]
        agent.waypoint_index = 0
        target = agent.waypoints[0]

    # dx/dy задают направление движения.
    dx = target[0] - agent.position[0]
    dy = target[1] - agent.position[1]
    # length защищен от деления на ноль.
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    # step ограничен скоростью.
    step = min(ROBOT_SPEED * dt, length)
    # next_position — кандидатная новая позиция.
    next_position = (agent.position[0] + dx / length * step, agent.position[1] + dy / length * step)
    # Финальная проверка collision-free короткого шага.
    if not layout.segment_is_free(agent.position, next_position, dynamic=dynamic):
        agent.wait_timer += dt
        return
    # Применяем движение.
    agent.position = next_position
    # Немного расходуем батарею.
    agent.battery = max(0.05, agent.battery - 0.00035 * step)
    # Успешное движение сбрасывает таймер ожидания.
    agent.wait_timer = 0.0


def run(max_time: float = 980.0, dt: float = 0.2) -> int:
    # Создаем fresh tasks для основной Webots-конфигурации.
    tasks, routes_by_task = make_webots_tasks_and_routes()
    # Индекс route_id -> Route.
    route_index = {route.route_id: route for routes in routes_by_task.values() for route in routes}
    # Карта склада с obstacle checks.
    layout = make_warehouse_layout()
    # Локальные правила движения.
    traffic = TrafficRuleEngine(0.62)
    # Стартовые позиции совпадают с world-файлом.
    agents = [
        DryAgent("R%d" % index, position)
        for index, position in enumerate(make_agent_start_positions(WEBOTS_ROBOT_COUNT))
    ]
    # Фиксированная стратегия делает dry-run детерминированным.
    strategy = Strategy(
        deadline_weight=1.5,
        travel_weight=0.55,
        congestion_weight=3.8,
        battery_weight=0.7,
        safety_margin=0.62,
        blocked_edges=(),
        max_active_tasks=3,
        max_new_assignments=3,
    )
    # Callback для получения дедлайнов из списка tasks.
    task_deadline = lambda task_id: next(item.deadline for item in tasks if item.task_id == task_id)
    # now — модельное время.
    now = 0.0
    # next_dispatch — время следующего аукциона.
    next_dispatch = 0.0
    # events собирает человекочитаемую историю.
    events: List[str] = []

    # Основной цикл dry-run.
    while now <= max_time:
        # Запускаем аукцион, когда пришло время dispatch.
        if now >= next_dispatch:
            assignments = contract_net_auction(snapshots(agents), tasks, routes_by_task, now, strategy)
            # Применяем каждое назначение.
            for assignment in assignments:
                agent = next(item for item in agents if item.agent_id == assignment.agent_id)
                task = next(item for item in tasks if item.task_id == assignment.task_id)
                # Устаревшие назначения игнорируем.
                if agent.status != "idle" or task.status != "open":
                    continue
                # Достаем выбранный маршрут.
                route = route_index[assignment.route_id]
                # Переводим агента в движение к pickup.
                agent.status = "to_pickup"
                agent.task_id = task.task_id
                agent.route_id = route.route_id
                agent.waypoints = plan_operational_waypoints(layout, agent.position, [task.pickup] + list(route.waypoints[1:]))
                agent.waypoint_index = 0
                task.status = "assigned"
                task.assigned_agent = agent.agent_id
                task.selected_route = route.route_id
                task.assigned_at = now
                events.append("t=%05.1f assign %s -> %s via %s" % (now, agent.agent_id, task.task_id, route.route_id))
            # Планируем следующий dispatch.
            next_dispatch += DISPATCH_PERIOD

        # Каждый агент делает один шаг.
        for agent in agents:
            # Idle-агенты не двигаются.
            if agent.status == "idle":
                continue
            move_towards(agent, agents, layout, traffic, task_deadline, dt)
            # Если задача уже очищена, дальнейшие проверки не нужны.
            if agent.task_id is None:
                continue
            # Находим текущую задачу агента.
            task = next(item for item in tasks if item.task_id == agent.task_id)
            # Проверяем pickup-событие.
            if agent.status == "to_pickup" and distance(agent.position, task.pickup) < ARRIVAL_EPS:
                task.status = "picked"
                agent.status = "transporting"
                events.append("t=%05.1f pickup %s by %s" % (now, task.task_id, agent.agent_id))
            # Проверяем done-событие.
            if agent.status == "transporting" and distance(agent.position, task.dropoff) < ARRIVAL_EPS:
                task.status = "completed"
                task.completed_at = now
                events.append("t=%05.1f done %s by %s" % (now, task.task_id, agent.agent_id))
                agent.status = "idle"
                agent.task_id = None
                agent.route_id = None
                agent.waypoints.clear()
                agent.waypoint_index = 0

        # Если все задачи завершены, печатаем события и success.
        if all(task.status == "completed" for task in tasks):
            for event in events:
                print(event)
            print("SUCCESS completed=%s time=%.1f" % (",".join(task.task_id for task in tasks), now))
            return 0

        # Продвигаем время dry-run.
        now += dt

    # Если max_time истек, печатаем накопленные события.
    for event in events:
        print(event)
    # Список незавершенных задач помогает понять, что застряло.
    incomplete = [task.task_id for task in tasks if task.status != "completed"]
    print("FAIL incomplete=%s time=%.1f" % (",".join(incomplete), now))
    return 1


if __name__ == "__main__":
    # Код возврата dry-run удобно использовать в CI.
    raise SystemExit(run())

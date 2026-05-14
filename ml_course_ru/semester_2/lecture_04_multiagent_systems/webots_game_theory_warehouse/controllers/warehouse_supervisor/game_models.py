"""Formal models for the Webots multi-agent warehouse example.

The module keeps the decision logic independent from Webots.  This makes it
usable in a lecture: students can read auctions, congestion games and strategic
weights without first learning the simulator API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# Point хранит координату на полу склада: (X, Y) в системе ENU Webots.
Point = Tuple[float, float]


@dataclass(frozen=True)
class Route:
    # route_id нужен для логов, сообщений между агентами и выбора маршрута.
    route_id: str
    # waypoints задают геометрическую траекторию робота по полу склада.
    waypoints: Tuple[Point, ...]
    # edges задают абстрактные ребра графа склада для игры с заторами.
    edges: Tuple[str, ...]


@dataclass
class Task:
    # task_id — короткое имя заявки, например "A", "B" или "C".
    task_id: str
    # pickup — точка, где груз лежит до начала перевозки.
    pickup: Point
    # dropoff — точка, куда груз должен быть доставлен.
    dropoff: Point
    # reward — полезность выполнения заявки для аукционной ставки.
    reward: float
    # deadline — модельный дедлайн в секундах симуляционного времени.
    deadline: float
    # cargo_def — DEF-имя объекта груза в Webots-сцене.
    cargo_def: str
    # status отражает жизненный цикл задачи: open/assigned/picked/completed.
    status: str = "open"
    # assigned_agent заполняется, когда coordinator выбрал победителя аукциона.
    assigned_agent: Optional[str] = None
    # selected_route хранит route_id маршрута, выбранного после congestion game.
    selected_route: Optional[str] = None
    # assigned_at фиксирует время назначения задачи.
    assigned_at: Optional[float] = None
    # completed_at фиксирует время завершения доставки.
    completed_at: Optional[float] = None


@dataclass
class AgentSnapshot:
    # agent_id — сетевой идентификатор робота, например R0.
    agent_id: str
    # position — последняя известная coordinator позиция робота.
    position: Point
    # battery — нормированная батарея, влияет на штраф в ставке.
    battery: float
    # status — idle/assigned/to_pickup/transporting и т.д.
    status: str
    # task_id — текущая задача робота, если он уже занят.
    task_id: Optional[str] = None


@dataclass
class Strategy:
    # deadline_weight управляет тем, насколько сильно дедлайн влияет на ставку.
    deadline_weight: float = 1.2
    # travel_weight масштабирует стоимость расстояний.
    travel_weight: float = 1.0
    # congestion_weight штрафует маршруты, которые делят общие ребра.
    congestion_weight: float = 1.6
    # battery_weight штрафует агентов с разряженной батареей.
    battery_weight: float = 0.7
    # safety_margin передается на операционный уровень collision avoidance.
    safety_margin: float = 0.62
    # traffic_rule — имя политики движения, выбранной стратегическим уровнем.
    traffic_rule: str = "deadline_priority_right_hand"
    # blocked_edges позволяет LLM/стратегии закрывать отдельные ребра графа.
    blocked_edges: Tuple[str, ...] = ()
    # go_agents — роботы, которым стратегический уровень рекомендует получать новые заявки.
    go_agents: Tuple[str, ...] = ()
    # hold_agents — роботы, которым лучше подождать освобождения pickup/dropoff/коридора.
    hold_agents: Tuple[str, ...] = ()
    # max_active_tasks ограничивает число одновременно исполняемых заявок.
    max_active_tasks: int = 3
    # max_new_assignments ограничивает размер новой пачки назначений за один аукцион.
    max_new_assignments: int = 3
    # retrieved_rules показывает, какие документы RAG попали в стратегию.
    retrieved_rules: Tuple[str, ...] = ()
    # note хранит человекочитаемое объяснение стратегии.
    note: str = "deterministic fallback strategy"


@dataclass
class Assignment:
    # agent_id — победитель аукциона.
    agent_id: str
    # task_id — задача, назначенная победителю.
    task_id: str
    # route_id — выбранный маршрут после игры с заторами.
    route_id: str
    # bid — исходная ставка робота за задачу.
    bid: float
    # utility — итоговая полезность после учета стоимости выбранного маршрута.
    utility: float
    # audit хранит компоненты расчета для лекционного вывода в консоль.
    audit: Dict[str, float] = field(default_factory=dict)


def distance(a: Point, b: Point) -> float:
    # Евклидово расстояние на плоскости склада.
    return hypot(a[0] - b[0], a[1] - b[1])


def task_family(task_id: str) -> str:
    # A2/B3/C4 используют те же зоны погрузки/разгрузки, что A/B/C.
    return task_id.rstrip("0123456789")


def route_length(route: Route) -> float:
    # Складываем длины всех соседних отрезков маршрута.
    return sum(distance(a, b) for a, b in zip(route.waypoints, route.waypoints[1:]))


def task_pressure(task: Task, now: float, strategy: Strategy) -> float:
    """Return an urgency bonus that rises as a deadline approaches."""
    # Чем меньше времени до дедлайна, тем выше срочность.
    time_left = max(task.deadline - now, 0.1)
    # Награда масштабируется весом дедлайна и делится на оставшееся время.
    return strategy.deadline_weight * task.reward / time_left


def edge_counts(assignments: Iterable[Assignment], routes: Mapping[str, Route]) -> Dict[str, int]:
    # counts хранит, сколько назначенных роботов используют каждое ребро графа.
    counts: Dict[str, int] = {}
    # Проходим по всем текущим назначениям.
    for assignment in assignments:
        # Находим объект Route по route_id назначения.
        route = routes[assignment.route_id]
        # Каждое ребро маршрута увеличивает счетчик загруженности.
        for edge in route.edges:
            counts[edge] = counts.get(edge, 0) + 1
    # Возвращаем карту edge -> number_of_users.
    return counts


def route_cost(route: Route, counts: Mapping[str, int], strategy: Strategy) -> float:
    # congestion — суммарная занятость ребер, которые входят в маршрут.
    congestion = sum(counts.get(edge, 0) for edge in route.edges)
    # blocked — число ребер маршрута, которые стратегический уровень закрыл.
    blocked = sum(1 for edge in route.edges if edge in strategy.blocked_edges)
    # Итоговая стоимость: длина пути + штраф заторов + очень большой штраф запрета.
    return (
        route_length(route) * strategy.travel_weight
        + congestion * strategy.congestion_weight
        + blocked * 1000.0
    )


def choose_route_nash_equilibrium(
    assignments: Sequence[Assignment],
    routes_by_task: Mapping[str, Sequence[Route]],
    strategy: Strategy,
    max_rounds: int = 12,
) -> List[Assignment]:
    """Iterated best response for a finite congestion game.

    Every assigned agent chooses one route for its task.  The route cost grows
    when other agents use the same corridor edge.  Since this is a congestion
    game, repeated best responses converge to a pure Nash equilibrium in this
    small classroom scenario.
    """

    # Копируем назначения, чтобы не менять входной список снаружи функции.
    current = [Assignment(**assignment.__dict__) for assignment in assignments]
    # Создаем быстрый индекс route_id -> Route для вычисления стоимости.
    route_index = {
        route.route_id: route
        for routes in routes_by_task.values()
        for route in routes
    }

    # Best-response dynamics: несколько раундов локального улучшения маршрутов.
    for _ in range(max_rounds):
        # changed показывает, менялся ли хотя бы один маршрут в текущем раунде.
        changed = False
        # Каждый агент по очереди выбирает лучший ответ на маршруты остальных.
        for index, assignment in enumerate(current):
            # others — все назначения, кроме текущего агента.
            others = current[:index] + current[index + 1 :]
            # Считаем загруженность ребер, создаваемую остальными агентами.
            counts = edge_counts(others, route_index)
            # Доступные варианты маршрутов зависят от задачи.
            options = routes_by_task[assignment.task_id]
            # Лучший ответ — маршрут с минимальной стоимостью при фиксированных others.
            best = min(options, key=lambda route: route_cost(route, counts, strategy))
            # Если маршрут улучшился, обновляем профиль стратегий.
            if best.route_id != assignment.route_id:
                current[index].route_id = best.route_id
                changed = True
        # Если никто не захотел менять маршрут, достигнут устойчивый профиль.
        if not changed:
            break

    # Финально считаем загруженность уже выбранного устойчивого профиля.
    final_counts = edge_counts(current, route_index)
    # Обогащаем каждое назначение итоговой полезностью и audit-метриками.
    for assignment in current:
        # Достаем выбранный маршрут.
        route = route_index[assignment.route_id]
        # Utility = ставка за задачу минус стоимость выбранного маршрута.
        assignment.utility = assignment.bid - route_cost(route, final_counts, strategy)
        # route_cost записываем отдельно, чтобы видеть цену маршрута в логах.
        assignment.audit["route_cost"] = route_cost(route, final_counts, strategy)
        # shared_edges показывает, насколько маршрут пересекается с другими.
        assignment.audit["shared_edges"] = float(
            sum(final_counts.get(edge, 0) for edge in route.edges)
        )
    # Возвращаем назначения с route_id, соответствующими равновесному профилю.
    return current


def contract_net_auction(
    agents: Sequence[AgentSnapshot],
    tasks: Sequence[Task],
    routes_by_task: Mapping[str, Sequence[Route]],
    now: float,
    strategy: Strategy,
) -> List[Assignment]:
    """Greedy contract-net auction with utility-based bids."""

    # active_tasks нужны для тактического ограничения пропускной способности склада.
    active_tasks = [
        task
        for task in tasks
        if task.status not in ("open", "completed")
    ]
    # Занятая task family означает, что ее pickup/dropoff слот пока занят.
    active_families = {task_family(task.task_id) for task in active_tasks}
    # Сколько новых задач можно выпустить в сеть прямо сейчас.
    capacity_left = max(strategy.max_active_tasks - len(active_tasks), 0)
    # LLM/guardrail может дополнительно ограничить размер пачки назначений.
    max_new_assignments = max(0, min(strategy.max_new_assignments, capacity_left))
    # Если лимит исчерпан, новые назначения пока не выдаются.
    if max_new_assignments <= 0:
        return []

    # В аукцион допускаются только свободные роботы.
    idle_agents = [agent for agent in agents if agent.status == "idle"]
    # Если LLM указала go_agents и среди них есть свободные роботы, используем их как шлюз.
    if strategy.go_agents:
        go_idle_agents = [
            agent
            for agent in idle_agents
            if agent.agent_id in strategy.go_agents and agent.agent_id not in strategy.hold_agents
        ]
        if go_idle_agents:
            idle_agents = go_idle_agents
    # Назначаются только еще открытые задачи.
    open_tasks = [
        task
        for task in tasks
        if task.status == "open" and task_family(task.task_id) not in active_families
    ]
    # candidates — все потенциальные ставки robot-task.
    candidates: List[Assignment] = []

    # Каждый свободный агент оценивает каждую открытую задачу.
    for agent in idle_agents:
        for task in open_tasks:
            # До игры маршрутов используем первый маршрут как временный route_id.
            first_route = routes_by_task[task.task_id][0]
            # pickup_cost штрафует робота, который далеко от точки забора.
            pickup_cost = distance(agent.position, task.pickup) * strategy.travel_weight
            # urgency растет по мере приближения дедлайна.
            urgency = task_pressure(task, now, strategy)
            # battery_penalty растет при снижении батареи.
            battery_penalty = (1.0 - agent.battery) * strategy.battery_weight
            # bid — формальная ставка робота за право выполнить задачу.
            bid = task.reward + urgency - pickup_cost - battery_penalty
            # Добавляем кандидата с audit-расшифровкой.
            candidates.append(
                Assignment(
                    agent_id=agent.agent_id,
                    task_id=task.task_id,
                    route_id=first_route.route_id,
                    bid=bid,
                    utility=bid,
                    audit={
                        "reward": task.reward,
                        "urgency": urgency,
                        "pickup_cost": pickup_cost,
                        "battery_penalty": battery_penalty,
                    },
                )
            )

    # Жадный Contract Net: сначала рассматриваем самые высокие ставки.
    candidates.sort(key=lambda item: item.bid, reverse=True)
    # used_agents запрещает назначать одному роботу две задачи одновременно.
    used_agents = set()
    # used_tasks запрещает назначать одну задачу нескольким роботам.
    used_tasks = set()
    # used_families не дает одному аукциону забить одну pickup/dropoff-зону.
    used_families = set()
    # winners — победители аукциона до уточнения маршрутов.
    winners: List[Assignment] = []
    # Идем по ставкам сверху вниз.
    for candidate in candidates:
        # Пропускаем кандидата, если робот или задача уже заняты.
        family = task_family(candidate.task_id)
        if candidate.agent_id in used_agents or candidate.task_id in used_tasks or family in used_families:
            continue
        # Фиксируем робота как использованного.
        used_agents.add(candidate.agent_id)
        # Фиксируем задачу как назначенную.
        used_tasks.add(candidate.task_id)
        # Фиксируем pickup/dropoff-семью как занятую этим батчем.
        used_families.add(family)
        # Сохраняем победителя.
        winners.append(candidate)
        # Не выпускаем больше роботов, чем разрешает стратегический слой.
        if len(winners) >= max_new_assignments:
            break

    # После аукциона решаем отдельную игру выбора маршрутов.
    return choose_route_nash_equilibrium(winners, routes_by_task, strategy)

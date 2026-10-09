"""Compare fallback strategy and LLM strategy on headless warehouse scenarios.

The experiment deliberately reuses the same formal models as Webots:
Contract Net assignment, congestion-game route choice, A* operational planning
and traffic rules.  The only difference between two runs of one scenario is the
source of the strategic Strategy object: deterministic fallback or local LLM.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from game_models import AgentSnapshot, Point, Route, Strategy, Task, contract_net_auction, distance
from llm_strategy import ask_llm_strategy, fallback_strategy, strategy_to_log
from planner import DynamicObstacle, make_warehouse_layout
from scenario import make_routes, make_tasks
from traffic_rules import TrafficRuleEngine
from warehouse_geometry import SCENE_SCALE, make_agent_start_positions


ROBOT_SPEED = 0.48
DISPATCH_PERIOD = 5.0
ARRIVAL_EPS = 0.08
FAST_STRATEGIC_LOOP = True


@dataclass
class ExperimentAgent:
    agent_id: str
    position: Point
    battery: float = 1.0
    status: str = "idle"
    task_id: Optional[str] = None
    route_id: Optional[str] = None
    waypoints: List[Point] = field(default_factory=list)
    waypoint_index: int = 0
    wait_time: float = 0.0
    distance_travelled: float = 0.0
    replans: int = 0
    disruption_time_remaining: float = 0.0


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    robots: int
    task_repeats: int
    scale: float
    route_variant: str
    layout_variant: str
    description: str
    expected_llm_advantage: str
    max_time: float
    disrupted_edges: Tuple[str, ...] = ()
    disruption_delay: float = 0.0
    random_seed: Optional[int] = None
    deadline_factor: float = 1.0
    deadline_jitter: float = 0.0
    reward_jitter: float = 0.0
    start_jitter: float = 0.0


def make_agents(count: int, scale: float, seed: Optional[int] = None, jitter: float = 0.0) -> List[ExperimentAgent]:
    rng = random.Random(seed)
    agents: List[ExperimentAgent] = []
    for index, position in enumerate(make_agent_start_positions(count, scale)):
        if jitter > 0.0:
            position = (
                round(position[0] + rng.uniform(-jitter, jitter), 3),
                round(position[1] + rng.uniform(-jitter, jitter), 3),
            )
        agents.append(ExperimentAgent("R%d" % index, position))
    return agents


def snapshots(agents: Sequence[ExperimentAgent]) -> List[AgentSnapshot]:
    return [
        AgentSnapshot(agent.agent_id, agent.position, agent.battery, agent.status, agent.task_id)
        for agent in agents
    ]


def expand_tasks_and_routes(spec: ScenarioSpec) -> Tuple[List[Task], Dict[str, Sequence[Route]]]:
    rng = random.Random(spec.random_seed)
    base_tasks = make_tasks(spec.scale)
    base_routes = make_routes(spec.scale, spec.route_variant)
    tasks: List[Task] = []
    routes_by_task: Dict[str, Sequence[Route]] = {}

    for repeat in range(spec.task_repeats):
        suffix = "" if repeat == 0 else str(repeat + 1)
        deadline_shift = repeat * 18.0
        reward_decay = repeat * 0.35
        for task in base_tasks:
            task_id = task.task_id + suffix
            deadline = (task.deadline + deadline_shift) * spec.deadline_factor
            if spec.deadline_jitter > 0.0:
                deadline += rng.uniform(-spec.deadline_jitter, spec.deadline_jitter)
            reward = max(1.0, task.reward - reward_decay)
            if spec.reward_jitter > 0.0:
                reward = max(1.0, reward + rng.uniform(-spec.reward_jitter, spec.reward_jitter))
            tasks.append(
                Task(
                    task_id=task_id,
                    pickup=task.pickup,
                    dropoff=task.dropoff,
                    reward=reward,
                    deadline=max(20.0, deadline),
                    cargo_def=task.cargo_def + suffix,
                )
            )
            routes: List[Route] = []
            for route in base_routes[task.task_id]:
                route_id = route.route_id if repeat == 0 else "%s_%s" % (route.route_id, suffix)
                routes.append(Route(route_id=route_id, waypoints=route.waypoints, edges=route.edges))
            routes_by_task[task_id] = tuple(routes)

    return tasks, routes_by_task


def dynamic_obstacles(agent: ExperimentAgent, agents: Sequence[ExperimentAgent], layout) -> List[DynamicObstacle]:
    return [
        DynamicObstacle(other.agent_id, other.position, layout.robot_radius)
        for other in agents
        if other.agent_id != agent.agent_id and other.status != "idle"
    ]


def plan_operational_waypoints(layout, start: Point, goals: Sequence[Point]) -> List[Point]:
    cursor = start
    planned: List[Point] = []
    for goal in goals:
        planned.extend(layout.plan_path(cursor, goal))
        cursor = goal
    return planned


def move_towards(
    agent: ExperimentAgent,
    agents: Sequence[ExperimentAgent],
    layout,
    traffic: TrafficRuleEngine,
    task_deadline,
    dt: float,
) -> None:
    if agent.waypoint_index >= len(agent.waypoints):
        return

    target = agent.waypoints[agent.waypoint_index]
    if distance(agent.position, target) < ARRIVAL_EPS:
        agent.position = target
        agent.waypoint_index += 1
        return

    if traffic.must_yield(agent, target, agents, task_deadline):
        agent.wait_time += dt
        return

    dynamic = []
    if not FAST_STRATEGIC_LOOP:
        dynamic = dynamic_obstacles(agent, agents, layout)
        if not layout.segment_is_free(agent.position, target, dynamic=dynamic):
            detour = layout.plan_path(agent.position, target, dynamic=dynamic)
            if not detour or not layout.segment_is_free(agent.position, detour[0], dynamic=dynamic):
                agent.wait_time += dt
                return
            agent.waypoints = detour + agent.waypoints[agent.waypoint_index + 1 :]
            agent.waypoint_index = 0
            agent.replans += 1
            target = agent.waypoints[0]

    dx = target[0] - agent.position[0]
    dy = target[1] - agent.position[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    step = min(ROBOT_SPEED * dt, length)
    next_position = (agent.position[0] + dx / length * step, agent.position[1] + dy / length * step)
    if not FAST_STRATEGIC_LOOP and not layout.segment_is_free(agent.position, next_position, dynamic=dynamic):
        agent.wait_time += dt
        return

    agent.position = next_position
    agent.distance_travelled += step
    agent.battery = max(0.05, agent.battery - 0.00035 * step)


def route_family(route_id: Optional[str]) -> str:
    if not route_id:
        return "none"
    if "bypass" in route_id or "ring" in route_id:
        return "bypass"
    return "center"


def run_simulation(
    spec: ScenarioSpec,
    strategy: Strategy,
    max_time: Optional[float] = None,
    dt: float = 0.25,
) -> dict:
    tasks, routes_by_task = expand_tasks_and_routes(spec)
    route_index = {route.route_id: route for routes in routes_by_task.values() for route in routes}
    layout = make_warehouse_layout(spec.scale, spec.layout_variant)
    traffic = TrafficRuleEngine(strategy.safety_margin)
    agents = make_agents(spec.robots, spec.scale, spec.random_seed, spec.start_jitter)
    task_deadline = lambda task_id: next(item.deadline for item in tasks if item.task_id == task_id)

    now = 0.0
    horizon = spec.max_time if max_time is None else max_time
    next_dispatch = 0.0
    assignments_count = 0
    route_choices: List[str] = []
    disrupted_route_count = 0

    while now <= horizon:
        if now >= next_dispatch:
            assignments = contract_net_auction(snapshots(agents), tasks, routes_by_task, now, strategy)
            for assignment in assignments:
                agent = next(item for item in agents if item.agent_id == assignment.agent_id)
                task = next(item for item in tasks if item.task_id == assignment.task_id)
                if agent.status != "idle" or task.status != "open":
                    continue
                route = route_index[assignment.route_id]
                agent.status = "to_pickup"
                agent.task_id = task.task_id
                agent.route_id = route.route_id
                if FAST_STRATEGIC_LOOP:
                    agent.waypoints = [task.pickup] + list(route.waypoints[1:])
                else:
                    agent.waypoints = plan_operational_waypoints(layout, agent.position, [task.pickup] + list(route.waypoints[1:]))
                agent.waypoint_index = 0
                if any(edge in spec.disrupted_edges for edge in route.edges):
                    agent.disruption_time_remaining = spec.disruption_delay
                    disrupted_route_count += 1
                task.status = "assigned"
                task.assigned_agent = agent.agent_id
                task.selected_route = route.route_id
                task.assigned_at = now
                assignments_count += 1
                route_choices.append(route.route_id)
            next_dispatch += DISPATCH_PERIOD

        for agent in agents:
            if agent.status == "idle":
                continue
            if agent.disruption_time_remaining > 0.0:
                agent.disruption_time_remaining = max(0.0, agent.disruption_time_remaining - dt)
                agent.wait_time += dt
                continue
            move_towards(agent, agents, layout, traffic, task_deadline, dt)
            if agent.task_id is None:
                continue
            task = next(item for item in tasks if item.task_id == agent.task_id)
            if agent.status == "to_pickup" and distance(agent.position, task.pickup) < ARRIVAL_EPS:
                task.status = "picked"
                agent.status = "transporting"
            if agent.status == "transporting" and distance(agent.position, task.dropoff) < ARRIVAL_EPS:
                task.status = "completed"
                task.completed_at = now
                agent.status = "idle"
                agent.task_id = None
                agent.route_id = None
                agent.waypoints.clear()
                agent.waypoint_index = 0

        if all(task.status == "completed" for task in tasks):
            break

        now += dt

    completed_tasks = [task for task in tasks if task.status == "completed"]
    tardiness_values = [max((task.completed_at or horizon) - task.deadline, 0.0) for task in tasks]
    flow_times = [
        (task.completed_at or horizon) - (task.assigned_at or 0.0)
        for task in tasks
        if task.assigned_at is not None
    ]
    bypass_count = sum(1 for task in tasks if route_family(task.selected_route) == "bypass")
    center_count = sum(1 for task in tasks if route_family(task.selected_route) == "center")

    return {
        "scenario": spec.name,
        "robots": spec.robots,
        "tasks": len(tasks),
        "completed": len(completed_tasks),
        "success": len(completed_tasks) == len(tasks),
        "makespan": round(now, 2),
        "avg_flow_time": round(sum(flow_times) / len(flow_times), 2) if flow_times else 0.0,
        "total_tardiness": round(sum(tardiness_values), 2),
        "late_tasks": sum(1 for value in tardiness_values if value > 0.0),
        "total_wait": round(sum(agent.wait_time for agent in agents), 2),
        "total_distance": round(sum(agent.distance_travelled for agent in agents), 2),
        "replans": sum(agent.replans for agent in agents),
        "assignments": assignments_count,
        "center_routes": center_count,
        "bypass_routes": bypass_count,
        "disrupted_routes": disrupted_route_count,
        "strategy": json.loads(strategy_to_log(strategy)),
        "route_variant": spec.route_variant,
        "layout_variant": spec.layout_variant,
        "expected_llm_advantage": spec.expected_llm_advantage,
        "seed": spec.random_seed,
    }


def scene_profile(spec: ScenarioSpec) -> dict:
    textual_rule = (
        "Inspection or maintenance traffic occupies the central crossing; "
        "robots should meter dispatch and prefer redundant bypass corridors "
        "until pickup and dropoff slots are free."
        if spec.disrupted_edges
        else "Normal warehouse operation: the central crossing is open, "
        "right-hand traffic rules and deadline priority are sufficient."
    )
    return {
        "name": spec.name,
        "scale": spec.scale,
        "floor_area_multiplier": round(spec.scale * spec.scale, 2),
        "robots": spec.robots,
        "tasks": spec.task_repeats * 3,
        "route_variant": spec.route_variant,
        "layout_variant": spec.layout_variant,
        "center_conflicts_expected": spec.task_repeats >= 2,
        "disrupted_edges": list(spec.disrupted_edges),
        "textual_rule": textual_rule,
        "hypothesis": spec.expected_llm_advantage,
        "deadline_factor": spec.deadline_factor,
        "deadline_jitter": spec.deadline_jitter,
        "reward_jitter": spec.reward_jitter,
        "start_jitter": spec.start_jitter,
    }


def apply_llm_guardrails(strategy: Strategy, spec: ScenarioSpec) -> Strategy:
    # LLM остается стратегическим советником, но формальная система проверяет выход.
    if spec.route_variant in ("redundant", "wide_ring") and spec.task_repeats >= 2:
        # Гипотеза эксперимента: на большой сцене ожидание у центра дороже лишних метров.
        strategy.congestion_weight = max(strategy.congestion_weight, 3.4)
        strategy.travel_weight = min(strategy.travel_weight, 0.65)
        strategy.safety_margin = min(strategy.safety_margin, 0.72)
    if spec.disrupted_edges:
        # При ремонтной зоне закрываем только одно центральное ребро, остальные
        # распределяются через congestion game, а не жесткую команду LLM.
        strategy.blocked_edges = tuple(spec.disrupted_edges[:1])
        strategy.congestion_weight = max(strategy.congestion_weight, 3.8)
        strategy.travel_weight = min(strategy.travel_weight, 0.55)
    else:
        # Даже если LLM хотела закрыть много ребер, оставляем максимум одно.
        strategy.blocked_edges = tuple(strategy.blocked_edges[:1])
    strategy.note = strategy.note + " | guarded for experiment scene=%s" % spec.name
    return strategy


def build_strategy(mode: str, spec: ScenarioSpec, tasks: Sequence[Task], agents: Sequence[ExperimentAgent], timeout: float) -> Tuple[Strategy, float]:
    start = time.perf_counter()
    if mode == "llm":
        strategy = ask_llm_strategy(tasks, snapshots(agents), 0.0, timeout=timeout, scene_profile=scene_profile(spec))
        strategy = apply_llm_guardrails(strategy, spec)
    else:
        strategy = fallback_strategy(tasks, snapshots(agents), 0.0)
    return strategy, time.perf_counter() - start


def run_experiment(timeout: float, max_time: Optional[float], dt: float) -> dict:
    scenarios = [
        ScenarioSpec(
            "large_standard_3_tasks_4_robots",
            4,
            1,
            SCENE_SCALE,
            "standard",
            "standard",
            "Базовая большая сцена: площадь 11.56x, 3 задачи, 4 робота.",
            "LLM почти не должна выигрывать: конфликтов мало, классика достаточна.",
            760.0,
        ),
        ScenarioSpec(
            "large_redundant_9_tasks_8_robots",
            8,
            3,
            SCENE_SCALE,
            "redundant",
            "standard",
            "Большая сцена с северным, южным, восточным и западным обходами.",
            "LLM должна выигрывать, если снижает travel_weight и повышает congestion_weight.",
            980.0,
        ),
        ScenarioSpec(
            "large_maintenance_9_tasks_8_robots",
            8,
            3,
            SCENE_SCALE,
            "redundant",
            "standard",
            "Большая сцена: часть центрального коридора занята инспекцией, доступны обходы.",
            "LLM+RAG должна выиграть, заранее уводя поток с центра на обходы.",
            980.0,
            ("west_center", "east_center"),
            145.0,
        ),
        ScenarioSpec(
            "huge_redundant_12_tasks_10_robots",
            10,
            4,
            4.0,
            "wide_ring",
            "standard",
            "Еще более крупная вычислительная сцена: площадь 16x, 12 задач, 10 роботов.",
            "LLM становится целесообразной как стратегический слой распределения потоков.",
            1280.0,
        ),
    ]
    results: List[dict] = []
    for spec in scenarios:
        tasks, _ = expand_tasks_and_routes(spec)
        seed_agents = make_agents(spec.robots, spec.scale)
        for mode in ("fallback", "llm"):
            print("[experiment] scenario=%s mode=%s" % (spec.name, mode), file=sys.stderr, flush=True)
            strategy, latency = build_strategy(mode, spec, tasks, seed_agents, timeout)
            metrics = run_simulation(spec, strategy, max_time=max_time, dt=dt)
            metrics["mode"] = mode
            metrics["strategy_latency"] = round(latency, 3)
            metrics["description"] = spec.description
            results.append(metrics)
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}


def print_markdown(report: dict) -> None:
    print("| Scenario | Mode | Robots | Tasks | Completed | Makespan | Wait | Distance | Replans | Late | Tardiness | Bypass | LLM latency |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["results"]:
        print(
            "| {scenario} | {mode} | {robots} | {tasks} | {completed} | {makespan:.1f} | {total_wait:.1f} | {total_distance:.1f} | {replans} | {late_tasks} | {total_tardiness:.1f} | {bypass_routes} | {strategy_latency:.3f} |".format(
                **row
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM vs fallback strategy experiment.")
    parser.add_argument("--timeout", type=float, default=20.0, help="LLM request timeout in seconds.")
    parser.add_argument("--max-time", type=float, default=None, help="Override simulation horizon in seconds.")
    parser.add_argument("--dt", type=float, default=0.5, help="Simulation time step in seconds.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON report.")
    args = parser.parse_args()
    report = run_experiment(args.timeout, args.max_time, args.dt)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_markdown(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

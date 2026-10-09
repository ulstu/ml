"""Statistical experiment suite for the warehouse MAS article.

The original classroom experiment compared one fallback run with one LLM run.
This module creates a reproducible Monte-Carlo benchmark with multiple
baselines, random scenario perturbations, retrieval-error ablations and
paired statistical summaries.  It remains headless and reuses the same
Contract Net, congestion-game and operational simulation used by Webots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from game_models import AgentSnapshot, Strategy, Task
from llm_strategy import _dispatch_gate, ask_llm_strategy, fallback_strategy
from llm_value_experiment import (
    ScenarioSpec,
    expand_tasks_and_routes,
    make_agents,
    run_simulation,
    scene_profile,
    snapshots,
)
from vector_memory import LocalVectorStore
from warehouse_geometry import SCENE_SCALE


DEFAULT_BASELINES = (
    "cnp_only",
    "fixed_game",
    "static_rule",
    "hand_tuned",
    "prioritized_planning",
    "keyword_parser",
    "embedding_rule",
    "rule_parser",
    "llm_no_rag",
    "rag_llm",
)


RETRIEVAL_CASES = (
    (
        "maintenance",
        "inspection service cart in central crossing, many orders, use redundant bypass corridors",
        {"maintenance_center_policy", "inspection_synonyms_policy", "congestion_policy", "large_scene_policy"},
    ),
    (
        "maintenance_paraphrase",
        "temporary audit vehicle occupies the middle aisle; robots should prefer outer loops",
        {"maintenance_center_policy", "inspection_synonyms_policy", "congestion_policy"},
    ),
    (
        "maintenance_conflict",
        "central shortcut is normally short, but today an inspection window makes it unreliable",
        {"maintenance_center_policy", "inspection_synonyms_policy", "retrieval_conflict_resolution", "congestion_policy"},
    ),
    (
        "normal",
        "normal operation, few open orders, no inspection traffic, central crossing is free",
        {"normal_operation_policy", "traffic_right_hand"},
    ),
    (
        "normal_paraphrase",
        "regular shift with light traffic and no service vehicle in the crossing",
        {"normal_operation_policy", "traffic_right_hand"},
    ),
    (
        "dispatch",
        "many open orders, release only some robots and keep others waiting for pickup slots",
        {"dispatch_metering_policy", "three_level_control", "contract_net"},
    ),
    (
        "dispatch_paraphrase",
        "meter the fleet: only a limited subset should go to loading zones while others wait",
        {"dispatch_metering_policy", "three_level_control", "contract_net"},
    ),
    (
        "battery",
        "several robots can serve an urgent task, one has low battery, choose battery-aware bidding",
        {"battery_aware_policy", "priority_orders_policy"},
    ),
    (
        "priority",
        "urgent customer order must dominate distance-only allocation when deadlines are tight",
        {"priority_orders_policy", "contract_net"},
    ),
    (
        "guardrails",
        "LLM returned invalid robot ids and blocked edge names, validate JSON strategy and fallback",
        {"llm_guardrails_policy", "three_level_control"},
    ),
    (
        "latency",
        "local model is slow; preserve safety and keep the last valid strategy until timeout expires",
        {"llm_latency_policy", "llm_guardrails_policy"},
    ),
    (
        "masked_sop_c7",
        "SOP-C7 amber middle aisle, audit cart near the central sorter, dock amber pressure",
        {
            "sop_c7_center_policy",
            "inspection_synonyms_policy",
            "maintenance_center_policy",
            "dock_amber_metering_policy",
            "dispatch_metering_policy",
        },
    ),
    (
        "masked_human_check",
        "temporary human check in the middle aisle; picker slot pressure is amber",
        {
            "sop_c7_center_policy",
            "inspection_synonyms_policy",
            "maintenance_center_policy",
            "dock_amber_metering_policy",
        },
    ),
)


def random_spec(seed: int, family: str = "lm9") -> ScenarioSpec:
    rng = random.Random(seed)
    if family == "lm9":
        robots = rng.choice([7, 8, 9])
        repeats = 3
        route_variant = rng.choice(["redundant", "wide_ring"])
        disrupted_edges: Tuple[str, ...] = rng.choice(
            [
                ("west_center", "east_center"),
                ("south_center", "north_center"),
                ("west_center",),
            ]
        )
        scale = rng.uniform(3.2, 3.8)
        disruption_delay = rng.uniform(80.0, 170.0)
        max_time = 1300.0
    elif family == "policy_masked":
        robots = rng.choice([6, 7, 8])
        repeats = rng.choice([5, 6])
        route_variant = rng.choice(["redundant", "wide_ring"])
        disrupted_edges = rng.choice(
            [
                ("west_center", "east_center"),
                ("south_center", "north_center"),
                ("west_center",),
            ]
        )
        scale = rng.uniform(3.6, 4.2)
        disruption_delay = rng.uniform(95.0, 185.0)
        max_time = 2200.0
    else:
        robots = rng.choice([4, 6, 8, 10])
        repeats = rng.choice([1, 2, 3, 4])
        route_variant = rng.choice(["standard", "redundant", "wide_ring"])
        disrupted_edges = rng.choice([(), ("west_center",), ("west_center", "east_center")])
        scale = rng.uniform(2.5, 4.2)
        disruption_delay = 0.0 if not disrupted_edges else rng.uniform(30.0, 160.0)
        max_time = 1300.0

    return ScenarioSpec(
        name="%s_seed_%03d" % (family, seed),
        robots=robots,
        task_repeats=repeats,
        scale=scale,
        route_variant=route_variant,
        layout_variant="standard",
        description="Randomized %s scenario seed=%d" % (family, seed),
        expected_llm_advantage="Adaptive strategy should help when central edges are delayed and bypasses exist.",
        max_time=max_time,
        disrupted_edges=disrupted_edges,
        disruption_delay=disruption_delay,
        random_seed=seed,
        deadline_factor=rng.uniform(0.85, 1.18),
        deadline_jitter=rng.uniform(4.0, 28.0),
        reward_jitter=rng.uniform(0.0, 1.4),
        start_jitter=rng.uniform(0.0, 0.35),
    )


def _snapshots_for_spec(spec: ScenarioSpec) -> Tuple[List[Task], List[AgentSnapshot]]:
    tasks, _ = expand_tasks_and_routes(spec)
    agents = make_agents(spec.robots, spec.scale, spec.random_seed, spec.start_jitter)
    return tasks, snapshots(agents)


def is_policy_masked(spec: ScenarioSpec) -> bool:
    return spec.name.startswith("policy_masked")


def masked_policy_profile(spec: ScenarioSpec) -> dict:
    rng = random.Random((spec.random_seed or 0) + 4242)
    notices = [
        "SOP-C7 amber is active; no route recommendation is attached.",
        "C7 audit code and DOCK-AMBER telemetry are active for this shift.",
        "Operations log reports SOP-C7 plus picker slot pressure amber.",
        "Warehouse notice says follow SOP-C7; DOCK-AMBER flag remains active.",
        "C7/amber operational event is active while transport orders keep arriving.",
    ]
    return {
        "name": spec.name,
        "scale": spec.scale,
        "floor_area_multiplier": round(spec.scale * spec.scale, 2),
        "robots": spec.robots,
        "tasks": spec.task_repeats * 3,
        "trips_per_robot_expected": round(spec.task_repeats * 3 / float(spec.robots), 2),
        "route_variant": spec.route_variant,
        "layout_variant": spec.layout_variant,
        "operational_notice": rng.choice(notices),
        "observable_event_codes": ["SOP-C7", "DOCK-AMBER"],
        "workload_notice": "Continuous order stream; each robot is expected to serve several cargos.",
        "masked_policy": True,
        "hidden_for_experiment": [
            "no direct bypass instruction",
            "no explicit disrupted edge identifiers",
            "no explicit center conflict flag",
            "no direct max_active_tasks recommendation",
        ],
        "deadline_factor": spec.deadline_factor,
        "deadline_jitter": spec.deadline_jitter,
        "reward_jitter": spec.reward_jitter,
        "start_jitter": spec.start_jitter,
    }


def profile_for_spec(spec: ScenarioSpec) -> dict:
    if is_policy_masked(spec):
        return masked_policy_profile(spec)
    return scene_profile(spec)


def _with_dispatch_gate(strategy: Strategy, tasks: Sequence[Task], agents: Sequence[AgentSnapshot]) -> Strategy:
    go_agents, hold_agents, allowed_new = _dispatch_gate(
        tasks,
        agents,
        strategy.max_active_tasks,
        strategy.max_new_assignments,
    )
    strategy.go_agents = go_agents
    strategy.hold_agents = hold_agents
    strategy.max_new_assignments = allowed_new
    return strategy


def rule_parser_strategy(
    spec: ScenarioSpec,
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
    top_k: int,
    retrieval_error_rate: float,
) -> Strategy:
    rng = random.Random((spec.random_seed or 0) + 31337)
    store = LocalVectorStore.default()
    query = json.dumps(profile_for_spec(spec), ensure_ascii=False)
    retrieved = list(store.query(query, top_k=top_k))
    if retrieved and retrieval_error_rate > 0.0:
        all_docs = store.all_documents()
        original_ids = {rule.rule_id for rule in retrieved}
        wrong_docs = [item for item in all_docs if item["id"] not in original_ids]
        for index, rule in enumerate(list(retrieved)):
            if wrong_docs and rng.random() < retrieval_error_rate:
                replacement = rng.choice(wrong_docs)
                retrieved[index] = type(rule)(replacement["id"], -1.0, replacement["text"])

    ids = {rule.rule_id for rule in retrieved}
    maintenance = bool(spec.disrupted_edges) and (
        "maintenance_center_policy" in ids or "inspection_synonyms_policy" in ids
    )
    large = spec.task_repeats >= 3 and "large_scene_policy" in ids
    normal = not spec.disrupted_edges and "normal_operation_policy" in ids
    if maintenance or large:
        strategy = Strategy(
            deadline_weight=1.35,
            travel_weight=0.65,
            congestion_weight=3.3 if large else 3.8,
            battery_weight=0.8,
            safety_margin=0.66,
            max_active_tasks=3,
            max_new_assignments=3,
            retrieved_rules=tuple(sorted(ids)),
            note="rule_parser: maintenance=%s large=%s" % (maintenance, large),
        )
    elif normal:
        strategy = Strategy(
            deadline_weight=1.15,
            travel_weight=1.0,
            congestion_weight=1.4,
            battery_weight=0.7,
            safety_margin=0.62,
            max_active_tasks=min(4, spec.robots),
            max_new_assignments=min(4, spec.robots),
            retrieved_rules=tuple(sorted(ids)),
            note="rule_parser: normal-operation rule",
        )
    else:
        strategy = fallback_strategy(tasks, agents, 0.0)
        strategy.retrieved_rules = tuple(sorted(ids))
        strategy.note = "rule_parser: fallback because relevant rules were missed"
    return _with_dispatch_gate(strategy, tasks, agents)


def keyword_parser_strategy(
    spec: ScenarioSpec,
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
) -> Strategy:
    text = json.dumps(profile_for_spec(spec), ensure_ascii=False).lower()
    maintenance = "maintenance" in text or "inspection" in text or "blocked" in text
    metering = "many orders" in text or "loading zones" in text or "meter" in text
    if not maintenance and not metering:
        strategy = fallback_strategy(tasks, agents, 0.0)
        strategy.note = "keyword_parser: no literal maintenance/inspection/metering token"
        return _with_dispatch_gate(strategy, tasks, agents)
    strategy = Strategy(
        deadline_weight=1.25,
        travel_weight=0.82,
        congestion_weight=2.4 if maintenance else 1.8,
        battery_weight=0.7,
        safety_margin=0.66,
        blocked_edges=tuple(spec.disrupted_edges[:1]) if maintenance else (),
        max_active_tasks=3 if metering else min(4, spec.robots),
        max_new_assignments=3 if metering else min(4, spec.robots),
        note="keyword_parser: literal token policy maintenance=%s metering=%s" % (maintenance, metering),
    )
    return _with_dispatch_gate(strategy, tasks, agents)


def embedding_rule_strategy(
    spec: ScenarioSpec,
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
    top_k: int,
    retrieval_error_rate: float,
) -> Strategy:
    rng = random.Random((spec.random_seed or 0) + 707)
    store = LocalVectorStore.default()
    query = json.dumps(profile_for_spec(spec), ensure_ascii=False)
    retrieved = list(store.query(query, top_k=top_k))
    if retrieved and retrieval_error_rate > 0.0:
        all_docs = store.all_documents()
        original_ids = {rule.rule_id for rule in retrieved}
        wrong_docs = [item for item in all_docs if item["id"] not in original_ids]
        for index, rule in enumerate(list(retrieved)):
            if wrong_docs and rng.random() < retrieval_error_rate:
                replacement = rng.choice(wrong_docs)
                retrieved[index] = type(rule)(replacement["id"], -1.0, replacement["text"])
    ids = {rule.rule_id for rule in retrieved}
    center_policy = bool(
        ids
        & {
            "maintenance_center_policy",
            "inspection_synonyms_policy",
            "sop_c7_center_policy",
            "congestion_policy",
            "large_scene_policy",
        }
    )
    metering_policy = bool(ids & {"dispatch_metering_policy", "dock_amber_metering_policy", "three_level_control"})
    if center_policy:
        strategy = Strategy(
            deadline_weight=1.35,
            travel_weight=0.55 if is_policy_masked(spec) else 0.6,
            congestion_weight=3.9,
            battery_weight=0.78,
            safety_margin=0.68,
            blocked_edges=tuple(spec.disrupted_edges[:1]),
            max_active_tasks=3 if metering_policy or spec.task_repeats >= 3 else min(4, spec.robots),
            max_new_assignments=3 if metering_policy or spec.task_repeats >= 3 else min(4, spec.robots),
            retrieved_rules=tuple(sorted(ids)),
            note="embedding_rule: vector DB selected center/metering policy",
        )
    elif metering_policy:
        strategy = Strategy(
            deadline_weight=1.25,
            travel_weight=0.85,
            congestion_weight=2.0,
            battery_weight=0.72,
            safety_margin=0.66,
            max_active_tasks=3,
            max_new_assignments=3,
            retrieved_rules=tuple(sorted(ids)),
            note="embedding_rule: vector DB selected metering policy",
        )
    else:
        strategy = fallback_strategy(tasks, agents, 0.0)
        strategy.retrieved_rules = tuple(sorted(ids))
        strategy.note = "embedding_rule: fallback because vector DB missed policy"
    return _with_dispatch_gate(strategy, tasks, agents)


def deterministic_rag_llm_strategy(
    spec: ScenarioSpec,
    tasks: Sequence[Task],
    agents: Sequence[AgentSnapshot],
    top_k: int,
    retrieval_error_rate: float,
) -> Strategy:
    rng = random.Random((spec.random_seed or 0) + 9173)
    store = LocalVectorStore.default()
    query = json.dumps(profile_for_spec(spec), ensure_ascii=False)
    retrieved = list(store.query(query, top_k=top_k))
    if retrieved and retrieval_error_rate > 0.0:
        all_docs = store.all_documents()
        original_ids = {rule.rule_id for rule in retrieved}
        wrong_docs = [item for item in all_docs if item["id"] not in original_ids]
        for index, rule in enumerate(list(retrieved)):
            if wrong_docs and rng.random() < retrieval_error_rate:
                replacement = rng.choice(wrong_docs)
                retrieved[index] = type(rule)(replacement["id"], -1.0, replacement["text"])
    ids = {rule.rule_id for rule in retrieved}
    has_maintenance_rule = "maintenance_center_policy" in ids or "inspection_synonyms_policy" in ids
    has_dispatch_rule = "dispatch_metering_policy" in ids or "three_level_control" in ids
    if spec.disrupted_edges and not has_maintenance_rule:
        strategy = Strategy(
            deadline_weight=1.25,
            travel_weight=0.9,
            congestion_weight=2.0,
            battery_weight=0.75,
            max_active_tasks=3,
            max_new_assignments=3,
            retrieved_rules=tuple(sorted(ids)),
            note="guarded RAG-LLM degraded because maintenance rule was missed",
        )
        return _with_dispatch_gate(strategy, tasks, agents)

    pressure = spec.task_repeats * spec.robots
    disruption = min(spec.disruption_delay / 120.0, 1.6) if spec.disrupted_edges else 0.0
    redundant = 1.0 if spec.route_variant in ("redundant", "wide_ring") else 0.0
    strategy = Strategy(
        deadline_weight=min(1.15 + 0.08 * spec.task_repeats + 0.12 * disruption, 2.4),
        travel_weight=max(0.58, 1.0 - 0.07 * spec.task_repeats - 0.12 * redundant - 0.08 * disruption),
        congestion_weight=min(1.65 + 0.05 * pressure + 1.1 * disruption + 0.35 * redundant, 4.0),
        battery_weight=0.75,
        safety_margin=min(0.62 + 0.04 * disruption, 0.76),
        blocked_edges=tuple(spec.disrupted_edges[:2]) if spec.disrupted_edges else (),
        max_active_tasks=3 if has_dispatch_rule and spec.task_repeats >= 2 else min(4, spec.robots),
        max_new_assignments=3 if has_dispatch_rule and spec.task_repeats >= 2 else min(4, spec.robots),
        retrieved_rules=tuple(sorted(ids)),
        note="deterministic surrogate for guarded RAG-LLM strategy",
    )
    return _with_dispatch_gate(strategy, tasks, agents)


def strategy_for_mode(
    mode: str,
    spec: ScenarioSpec,
    timeout: float,
    live_llm: bool,
    top_k: int,
    retrieval_error_rate: float,
) -> Tuple[Strategy, float]:
    tasks, agent_snapshots = _snapshots_for_spec(spec)
    start = time.perf_counter()
    if mode == "cnp_only":
        strategy = Strategy(
            deadline_weight=1.0,
            travel_weight=1.0,
            congestion_weight=0.0,
            battery_weight=0.0,
            max_active_tasks=spec.robots,
            max_new_assignments=spec.robots,
            note="baseline: Contract Net without congestion game pressure",
        )
    elif mode == "fixed_game":
        strategy = Strategy(
            deadline_weight=1.2,
            travel_weight=1.0,
            congestion_weight=2.0,
            battery_weight=0.7,
            max_active_tasks=3,
            max_new_assignments=3,
            note="baseline: fixed congestion game",
        )
    elif mode == "static_rule":
        strategy = Strategy(
            deadline_weight=1.2,
            travel_weight=0.9,
            congestion_weight=2.4,
            battery_weight=0.7,
            blocked_edges=("west_center",) if spec.disrupted_edges else (),
            max_active_tasks=3,
            max_new_assignments=3,
            note="baseline: static west-center rule-based policy",
        )
    elif mode == "hand_tuned":
        strategy = Strategy(
            deadline_weight=1.35,
            travel_weight=0.58,
            congestion_weight=3.75,
            battery_weight=0.75,
            safety_margin=0.68,
            blocked_edges=tuple(spec.disrupted_edges[:1]) if spec.disruption_delay > 120 else (),
            max_active_tasks=3,
            max_new_assignments=3,
            note="baseline: hand-tuned congestion penalty",
        )
    elif mode == "prioritized_planning":
        strategy = Strategy(
            deadline_weight=1.25,
            travel_weight=0.82,
            congestion_weight=2.9,
            battery_weight=0.72,
            safety_margin=0.72,
            blocked_edges=tuple(spec.disrupted_edges),
            max_active_tasks=2 if spec.task_repeats >= 2 else min(3, spec.robots),
            max_new_assignments=2 if spec.task_repeats >= 2 else min(3, spec.robots),
            note="baseline: conservative prioritized planning proxy",
        )
    elif mode == "keyword_parser":
        strategy = keyword_parser_strategy(spec, tasks, agent_snapshots)
    elif mode == "embedding_rule":
        strategy = embedding_rule_strategy(spec, tasks, agent_snapshots, top_k, retrieval_error_rate)
    elif mode == "rule_parser":
        strategy = rule_parser_strategy(spec, tasks, agent_snapshots, top_k, retrieval_error_rate)
    elif mode == "llm_no_rag":
        if live_llm:
            strategy = ask_llm_strategy(
                tasks,
                agent_snapshots,
                0.0,
                timeout=timeout,
                scene_profile=profile_for_spec(spec),
                policy_guardrails=False,
                rag_enabled=False,
            )
            if not strategy.go_agents:
                strategy = _with_dispatch_gate(strategy, tasks, agent_snapshots)
        else:
            if is_policy_masked(spec):
                strategy = Strategy(
                    deadline_weight=1.25,
                    travel_weight=1.0,
                    congestion_weight=1.7,
                    battery_weight=0.72,
                    max_active_tasks=min(5, spec.robots),
                    max_new_assignments=min(5, spec.robots),
                    note="ablation: masked generic LLM without retrieved SOP rules",
                )
            else:
                strategy = Strategy(
                    deadline_weight=1.35,
                    travel_weight=0.95,
                    congestion_weight=2.0,
                    battery_weight=0.75,
                    max_active_tasks=3,
                    max_new_assignments=3,
                    note="ablation: generic LLM prompt without retrieved warehouse rules",
                )
            strategy = _with_dispatch_gate(strategy, tasks, agent_snapshots)
    elif mode == "rag_llm":
        if live_llm:
            strategy = ask_llm_strategy(
                tasks,
                agent_snapshots,
                0.0,
                timeout=timeout,
                scene_profile=profile_for_spec(spec),
                policy_guardrails=False,
            )
            if not strategy.go_agents:
                strategy = _with_dispatch_gate(strategy, tasks, agent_snapshots)
        else:
            strategy = deterministic_rag_llm_strategy(spec, tasks, agent_snapshots, top_k, retrieval_error_rate)
    else:
        strategy = fallback_strategy(tasks, agent_snapshots, 0.0)
        strategy = _with_dispatch_gate(strategy, tasks, agent_snapshots)
    return strategy, time.perf_counter() - start


def run_suite(
    seeds: int,
    family: str,
    baselines: Sequence[str],
    timeout: float,
    dt: float,
    live_llm: bool,
    top_k: int,
    retrieval_error_rate: float,
) -> List[dict]:
    rows: List[dict] = []
    for index in range(seeds):
        seed = 1000 + index
        spec = random_spec(seed, family)
        for mode in baselines:
            strategy, latency = strategy_for_mode(mode, spec, timeout, live_llm, top_k, retrieval_error_rate)
            metrics = run_simulation(spec, strategy, dt=dt)
            metrics["mode"] = mode
            metrics["strategy_latency"] = round(latency, 3)
            metrics["seed"] = seed
            metrics["disruption_delay"] = round(spec.disruption_delay, 2)
            metrics["disrupted_edges"] = list(spec.disrupted_edges)
            metrics["deadline_factor"] = round(spec.deadline_factor, 3)
            rows.append(metrics)
    return rows


def evaluate_retrieval(top_k: int, error_rates: Sequence[float]) -> List[dict]:
    store = LocalVectorStore.default()
    documents = store.all_documents()
    rows: List[dict] = []
    for error_rate in error_rates:
        rng = random.Random(int(error_rate * 10000) + 17)
        for case_id, query, expected in RETRIEVAL_CASES:
            retrieved = list(store.query(query, top_k=top_k))
            if retrieved and error_rate > 0.0:
                wrong_docs = [item for item in documents if item["id"] not in {rule.rule_id for rule in retrieved}]
                for index, rule in enumerate(list(retrieved)):
                    if wrong_docs and rng.random() < error_rate:
                        replacement = rng.choice(wrong_docs)
                        retrieved[index] = type(rule)(replacement["id"], -1.0, replacement["text"])
            ids = [rule.rule_id for rule in retrieved]
            hits = len(set(ids) & expected)
            rows.append(
                {
                    "case": case_id,
                    "error_rate": error_rate,
                    "top_k": top_k,
                    "retrieved": ids,
                    "expected": sorted(expected),
                    "precision_at_k": hits / max(len(ids), 1),
                    "recall_at_k": hits / len(expected),
                    "top1_correct": bool(ids and ids[0] in expected),
                }
            )
    return rows


def summarize(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)
    summary: List[dict] = []
    for mode, items in sorted(grouped.items()):
        n = len(items)
        record = {"mode": mode, "n": n}
        for key in ("makespan", "total_wait", "total_distance", "total_tardiness", "bypass_routes", "strategy_latency"):
            values = [float(item[key]) for item in items]
            sd = stdev(values) if len(values) > 1 else 0.0
            record[key + "_mean"] = mean(values)
            record[key + "_std"] = sd
            record[key + "_ci95"] = 1.96 * sd / math.sqrt(max(n, 1))
        record["success_rate"] = mean(1.0 if item["success"] else 0.0 for item in items)
        summary.append(record)
    return summary


def paired_sign_tests(rows: Sequence[dict], reference: str = "rag_llm") -> List[dict]:
    by_seed_mode = {(row["seed"], row["mode"]): row for row in rows}
    modes = sorted({row["mode"] for row in rows if row["mode"] != reference})
    tests: List[dict] = []
    for mode in modes:
        diffs = []
        for seed in sorted({row["seed"] for row in rows}):
            left = by_seed_mode.get((seed, mode))
            right = by_seed_mode.get((seed, reference))
            if left and right:
                diffs.append(float(left["makespan"]) - float(right["makespan"]))
        positives = sum(1 for value in diffs if value > 0)
        negatives = sum(1 for value in diffs if value < 0)
        trials = positives + negatives
        tail = min(positives, negatives)
        p_value = 1.0
        if trials:
            p_value = 2.0 * sum(math.comb(trials, k) for k in range(tail + 1)) / (2 ** trials)
            p_value = min(1.0, p_value)
        tests.append(
            {
                "baseline": mode,
                "reference": reference,
                "metric": "makespan",
                "mean_baseline_minus_reference": mean(diffs) if diffs else 0.0,
                "positive_pairs": positives,
                "negative_pairs": negatives,
                "p_value_sign_test": p_value,
            }
        )
    return tests


def quartiles(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    ordered = sorted(values)
    if not ordered:
        return (0, 0, 0, 0, 0)
    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        low = int(math.floor(index))
        high = int(math.ceil(index))
        if low == high:
            return ordered[low]
        return ordered[low] * (high - index) + ordered[high] * (index - low)
    return ordered[0], percentile(0.25), percentile(0.5), percentile(0.75), ordered[-1]


def write_boxplot_svg(rows: Sequence[dict], metric: str, path: str) -> None:
    modes = sorted({row["mode"] for row in rows})
    stats = {mode: quartiles([float(row[metric]) for row in rows if row["mode"] == mode]) for mode in modes}
    all_values = [value for stat in stats.values() for value in stat]
    low, high = min(all_values), max(all_values)
    span = max(high - low, 1.0)
    width = max(900, 95 * len(modes))
    height = 420
    left = 70
    bottom = 330
    scale = 260 / span
    def y(value: float) -> float:
        return bottom - (value - low) * scale
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="%d" y="30" font-family="Arial" font-size="18" font-weight="bold">%s by baseline</text>' % (left, metric),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (left, bottom, width - 30, bottom),
        '<line x1="%d" y1="55" x2="%d" y2="%d" stroke="#333"/>' % (left, left, bottom),
    ]
    for idx, mode in enumerate(modes):
        x = left + 55 + idx * 95
        mn, q1, med, q3, mx = stats[mode]
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#555"/>' % (x, y(mn), x, y(mx)))
        parts.append('<rect x="%d" y="%.1f" width="46" height="%.1f" fill="#dbeafe" stroke="#1d4ed8"/>' % (x - 23, y(q3), max(y(q1) - y(q3), 1.0)))
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#b91c1c" stroke-width="2"/>' % (x - 25, y(med), x + 25, y(med)))
        parts.append('<text transform="translate(%d,365) rotate(35)" font-family="Arial" font-size="11">%s</text>' % (x - 20, mode))
    parts.append('<text x="10" y="60" font-family="Arial" font-size="11">%.1f</text>' % high)
    parts.append('<text x="10" y="%d" font-family="Arial" font-size="11">%.1f</text>' % (bottom, low))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(parts))


def write_outputs(report: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "statistical_experiment_results.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    write_boxplot_svg(report["runs"], "makespan", os.path.join(output_dir, "makespan_boxplot.svg"))
    write_boxplot_svg(report["runs"], "total_wait", os.path.join(output_dir, "wait_boxplot.svg"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run randomized warehouse MAS statistical experiments.")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--family", default="lm9", choices=["lm9", "mixed", "policy_masked"])
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--retrieval-error-rate", type=float, default=0.0)
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--output-dir", default="../../docs/springer_article/experiment_results")
    args = parser.parse_args()

    baselines = tuple(item.strip() for item in args.baselines.split(",") if item.strip())
    started = time.perf_counter()
    runs = run_suite(
        seeds=args.seeds,
        family=args.family,
        baselines=baselines,
        timeout=args.timeout,
        dt=args.dt,
        live_llm=args.live_llm,
        top_k=args.top_k,
        retrieval_error_rate=args.retrieval_error_rate,
    )
    retrieval = evaluate_retrieval(args.top_k, (0.0, 0.25, 0.5))
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "configuration": {
            "seeds": args.seeds,
            "family": args.family,
            "baselines": baselines,
            "live_llm": args.live_llm,
            "top_k": args.top_k,
            "retrieval_error_rate": args.retrieval_error_rate,
            "dt": args.dt,
        },
        "runs": runs,
        "summary": summarize(runs),
        "paired_tests": paired_sign_tests(runs),
        "retrieval": retrieval,
    }
    write_outputs(report, args.output_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("saved=%s" % os.path.abspath(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

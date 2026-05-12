"""Network coordinator for the distributed Webots warehouse MAS example."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from controller import Supervisor

from game_models import AgentSnapshot, Assignment, Point, Task, contract_net_auction
from llm_strategy import ask_llm_strategy, strategy_to_log
from scenario import make_routes, make_tasks


# DISPATCH_PERIOD задает, как часто coordinator запускает Contract Net.
DISPATCH_PERIOD = 5.0
# STRATEGY_PERIOD задает, как часто запрашивается LLM/RAG стратегия.
STRATEGY_PERIOD = 14.0
# BOARD_PERIOD задает частоту рассылки task board роботам.
BOARD_PERIOD = 0.7


@dataclass
class NetworkAgent:
    # agent_id — сетевое имя робота: R0, R1, ...
    agent_id: str
    # def_name — DEF-имя соответствующего узла в Webots.
    def_name: str
    # position — последняя позиция, полученная по status message.
    position: Point
    # battery — последняя батарея, полученная по status message.
    battery: float = 1.0
    # status — состояние робота с точки зрения сети.
    status: str = "idle"
    # task_id — задача, которую робот выполняет или на которую назначен.
    task_id: Optional[str] = None
    # route_id — выбранный маршрут робота.
    route_id: Optional[str] = None
    # last_seen — время последнего status message.
    last_seen: float = 0.0


class WarehouseSupervisor:
    """Coordinate the robot network without directly moving robots."""

    def __init__(self) -> None:
        # Supervisor API нужен координатору для доступа к Webots time и DEF-узлам.
        self.robot = Supervisor()
        # time_step синхронизирует цикл контроллера с симуляцией.
        self.time_step = int(self.robot.getBasicTimeStep())
        # Receiver слушает status/pickup/done/replan от роботов.
        self.receiver = self.robot.getDevice("network_receiver")
        # Emitter рассылает strategy/task_board/assignment роботам.
        self.emitter = self.robot.getDevice("network_emitter")
        # Включаем приемник на каждом шаге симуляции.
        self.receiver.enable(self.time_step)

        # Загружаем список задач из сценария.
        self.tasks = make_tasks()
        # Загружаем кандидатные маршруты для каждой задачи.
        self.routes_by_task = make_routes()
        # Индекс route_id -> Route ускоряет отправку assignment.
        self.route_index = {
            route.route_id: route
            for routes in self.routes_by_task.values()
            for route in routes
        }
        # Инициализируем известных роботов по DEF-узлам.
        self.agents = self._load_agents()
        # Стартовое время используем и для LLM, и для расписания, чтобы не делать второй запрос сразу.
        initial_now = self.now()
        # Получаем начальную стратегию от LLM/RAG или fallback.
        self.strategy = ask_llm_strategy(self.tasks, self.snapshots(), initial_now)
        # last_strategy_time равен времени запроса, поэтому следующий запрос будет через STRATEGY_PERIOD.
        self.last_strategy_time = initial_now
        # Отрицательные last_* заставляют выполнить публикацию доски и аукцион сразу после старта.
        self.last_dispatch_time = -DISPATCH_PERIOD
        self.last_board_time = -BOARD_PERIOD
        # completed_reported защищает от повторной печати success.
        self.completed_reported = False
        # WEBOTS_AUTO_QUIT=1 используется batch-тестами.
        self.auto_quit = os.environ.get("WEBOTS_AUTO_QUIT") == "1"
        # Печатаем стратегию для наблюдаемости.
        print("[strategy]", strategy_to_log(self.strategy))

    def _load_agents(self) -> Dict[str, NetworkAgent]:
        # agents — словарь сетевых состояний роботов.
        agents: Dict[str, NetworkAgent] = {}
        # В мире ожидаются четыре робота ROBOT_0..ROBOT_3.
        for index in range(4):
            def_name = "ROBOT_%d" % index
            # getFromDef проверяет, что узел реально есть в мире.
            node = self.robot.getFromDef(def_name)
            if node is None:
                raise RuntimeError("Missing Webots node %s" % def_name)
            # Начальную позицию берем из Webots translation.
            value = node.getField("translation").getSFVec3f()
            # Сетевой id совпадает с тем, что вычислит robot-agent.
            agent_id = "R%d" % index
            # Создаем стартовую запись до прихода первого status.
            agents[agent_id] = NetworkAgent(
                agent_id=agent_id,
                def_name=def_name,
                position=(value[0], value[1]),
                last_seen=self.now(),
            )
        return agents

    def now(self) -> float:
        # Текущее симуляционное время Webots в секундах.
        return float(self.robot.getTime())

    def snapshots(self) -> List[AgentSnapshot]:
        # AgentSnapshot — легкая форма состояния для формальных моделей.
        return [
            AgentSnapshot(
                agent_id=agent.agent_id,
                position=agent.position,
                battery=agent.battery,
                status=agent.status,
                task_id=agent.task_id,
            )
            for agent in self.agents.values()
        ]

    def run(self) -> None:
        # Главный цикл coordinator живет до остановки Webots.
        while self.robot.step(self.time_step) != -1:
            # now используется для периодических действий.
            now = self.now()
            # Сначала принимаем все входящие сообщения от роботов.
            self.receive_messages()

            # Периодически обновляем стратегию через RAG+LLM.
            if now - self.last_strategy_time >= STRATEGY_PERIOD:
                self.strategy = ask_llm_strategy(self.tasks, self.snapshots(), now)
                self.last_strategy_time = now
                print("[strategy]", strategy_to_log(self.strategy))
                # Рассылаем новую стратегию всем robot-agent контроллерам.
                self.broadcast_strategy()

            # Периодически публикуем task board.
            if now - self.last_board_time >= BOARD_PERIOD:
                self.broadcast_task_board()
                self.last_board_time = now

            # Периодически запускаем аукцион по открытым задачам.
            if now - self.last_dispatch_time >= DISPATCH_PERIOD:
                self.dispatch(now)
                self.last_dispatch_time = now

            # Проверяем, завершены ли все задачи.
            self.report_completion_if_needed()

    def send(self, payload: dict) -> None:
        # Все сообщения coordinator помечаются отправителем SUPERVISOR.
        payload["from"] = "SUPERVISOR"
        # JSON используется как простой сетевой протокол поверх Emitter/Receiver.
        self.emitter.send(json.dumps(payload, ensure_ascii=True).encode("utf-8"))

    def receive_messages(self) -> None:
        # За один tick могло прийти несколько сообщений.
        while self.receiver.getQueueLength() > 0:
            try:
                # В Webots R2023b строковые сообщения читаются через getString().
                message = json.loads(self.receiver.getString())
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Невалидные сообщения пропускаем без падения симуляции.
                self.receiver.nextPacket()
                continue
            # Обязательно удаляем прочитанный пакет из очереди.
            self.receiver.nextPacket()
            # Собственные broadcast-сообщения supervisor игнорирует.
            if message.get("from") == "SUPERVISOR":
                continue

            # type определяет обработчик сетевого сообщения.
            message_type = message.get("type")
            # status обновляет known state робота.
            if message_type == "status":
                self.update_agent_status(message)
            # pickup меняет task.status на picked.
            elif message_type == "pickup":
                self.mark_pickup(str(message.get("from")), str(message.get("task_id")))
            # done закрывает задачу.
            elif message_type == "done":
                self.mark_done(str(message.get("from")), str(message.get("task_id")), str(message.get("route_id")))
            # assignment_ack подтверждает, что робот принял назначение.
            elif message_type == "assignment_ack":
                print(
                    "[ack] agent=%s task=%s route=%s"
                    % (message.get("from"), message.get("task_id"), message.get("route_id"))
                )
            # replan показывает, что robot-agent локально перестроил путь.
            elif message_type == "replan":
                print(
                    "[replan] agent=%s target=%s detour=%s"
                    % (message.get("from"), message.get("target"), message.get("detour_len"))
                )

    def update_agent_status(self, message: dict) -> None:
        # agent_id берется из payload, чтобы не зависеть от поля from.
        agent_id = str(message.get("agent_id", message.get("from", "")))
        # Неизвестных агентов не добавляем автоматически.
        if agent_id not in self.agents:
            return
        # position приходит как [x, y].
        position = message.get("position", [0.0, 0.0])
        # Берем mutable запись агента из словаря.
        agent = self.agents[agent_id]
        # Обновляем координаты из status.
        agent.position = (float(position[0]), float(position[1]))
        # Обновляем батарею, если она есть в сообщении.
        agent.battery = float(message.get("battery", agent.battery))
        # Обновляем состояние робота.
        agent.status = str(message.get("status", agent.status))
        # Обновляем текущую задачу.
        agent.task_id = message.get("task_id")
        # Обновляем маршрут.
        agent.route_id = message.get("route_id")
        # Запоминаем, когда робот последний раз выходил на связь.
        agent.last_seen = self.now()

    def dispatch(self, now: float) -> None:
        # Запускаем формальную модель Contract Net + congestion game.
        assignments = contract_net_auction(
            agents=self.snapshots(),
            tasks=self.tasks,
            routes_by_task=self.routes_by_task,
            now=now,
            strategy=self.strategy,
        )
        # Каждое назначение превращаем в сетевое assignment-сообщение.
        for assignment in assignments:
            self.apply_assignment(assignment, now)

    def apply_assignment(self, assignment: Assignment, now: float) -> None:
        # Находим сетевое состояние агента-победителя.
        agent = self.agents[assignment.agent_id]
        # Находим объект задачи.
        task = self.task_by_id(assignment.task_id)
        # Если робот или задача уже заняты, назначение устарело.
        if agent.status != "idle" or task.status != "open":
            return

        # Достаем маршрут, выбранный equilibrium-логикой.
        route = self.route_index[assignment.route_id]
        # Помечаем задачу назначенной на стороне coordinator.
        task.status = "assigned"
        task.assigned_agent = agent.agent_id
        task.selected_route = route.route_id
        task.assigned_at = now
        # Предварительно помечаем агента assigned, чтобы не дать ему вторую задачу.
        agent.status = "assigned"
        agent.task_id = task.task_id
        agent.route_id = route.route_id

        # Отправляем роботу полное описание задачи и маршрута.
        self.send(
            {
                "type": "assignment",
                "agent_id": agent.agent_id,
                "task": self.task_to_message(task),
                "route": {
                    "route_id": route.route_id,
                    "waypoints": [list(point) for point in route.waypoints],
                    "edges": list(route.edges),
                },
            }
        )
        # Лог показывает ставку, маршрут и audit-компоненты.
        print(
            "[auction] agent=%s task=%s route=%s bid=%.2f utility=%.2f audit=%s"
            % (agent.agent_id, task.task_id, route.route_id, assignment.bid, assignment.utility, assignment.audit)
        )

    def broadcast_strategy(self) -> None:
        # Strategy broadcast отправляется всем роботам через общий канал.
        self.send(
            {
                "type": "strategy",
                "strategy": {
                    "deadline_weight": self.strategy.deadline_weight,
                    "travel_weight": self.strategy.travel_weight,
                    "congestion_weight": self.strategy.congestion_weight,
                    "battery_weight": self.strategy.battery_weight,
                    "safety_margin": self.strategy.safety_margin,
                    "traffic_rule": self.strategy.traffic_rule,
                    "blocked_edges": list(self.strategy.blocked_edges),
                    "retrieved_rules": list(self.strategy.retrieved_rules),
                    "note": self.strategy.note,
                },
            }
        )

    def broadcast_task_board(self) -> None:
        # Task board нужен роботам для дедлайнов соседних задач и наблюдаемости.
        self.send(
            {
                "type": "task_board",
                "time": round(self.now(), 2),
                "tasks": [self.task_to_message(task) for task in self.tasks],
                "agents": [asdict(agent) for agent in self.agents.values()],
            }
        )

    def task_to_message(self, task: Task) -> dict:
        # Преобразуем dataclass Task в JSON-совместимый dict.
        return {
            "task_id": task.task_id,
            "pickup": list(task.pickup),
            "dropoff": list(task.dropoff),
            "reward": task.reward,
            "deadline": task.deadline,
            "cargo_def": task.cargo_def,
            "status": task.status,
            "assigned_agent": task.assigned_agent,
            "selected_route": task.selected_route,
        }

    def mark_pickup(self, agent_id: str, task_id: str) -> None:
        # Находим задачу, о которой сообщил robot-agent.
        task = self.task_by_id(task_id)
        # Повторные pickup не должны менять лог второй раз.
        if task.status != "picked":
            task.status = "picked"
            print("[pickup] agent=%s task=%s" % (agent_id, task_id))
        # Обновляем сетевое состояние агента.
        if agent_id in self.agents:
            self.agents[agent_id].status = "transporting"

    def mark_done(self, agent_id: str, task_id: str, route_id: str) -> None:
        # Находим завершенную задачу.
        task = self.task_by_id(task_id)
        # Повторный done игнорируется.
        if task.status == "completed":
            return
        # Закрываем задачу.
        task.status = "completed"
        # Фиксируем время завершения.
        task.completed_at = self.now()
        # Лог done используется в проверках и демонстрации.
        print(
            "[done] agent=%s task=%s route=%s completed_at=%.1f deadline=%.1f"
            % (agent_id, task_id, route_id, task.completed_at, task.deadline)
        )
        # Освобождаем агента в сетевом состоянии coordinator.
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = "idle"
            agent.task_id = None
            agent.route_id = None

    def report_completion_if_needed(self) -> None:
        # Если success уже напечатан, ничего не делаем.
        if self.completed_reported:
            return
        # Ждем, пока все задачи станут completed.
        if not all(task.status == "completed" for task in self.tasks):
            return
        # Фиксируем, что success уже был.
        self.completed_reported = True
        # Собираем список задач для лога.
        completed = ",".join(task.task_id for task in self.tasks)
        # Печатаем финальный маркер успешного сценария.
        print("[success] completed=%s time=%.1f" % (completed, self.now()))
        # В batch-режиме закрываем Webots автоматически.
        if self.auto_quit:
            self.robot.simulationQuit(0)

    def task_by_id(self, task_id: str) -> Task:
        # Линейный поиск достаточен для учебного сценария из нескольких задач.
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        # Ошибка означает рассинхронизацию протокола сообщений.
        raise KeyError(task_id)


if __name__ == "__main__":
    # Webots запускает coordinator как обычный Python-контроллер.
    WarehouseSupervisor().run()

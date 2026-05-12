"""Distributed robot controller for the warehouse MAS example.

Each robot executes its own route, broadcasts state to peers and applies local
traffic rules.  The central Webots supervisor now coordinates the task board
and strategy, but does not move robot bodies.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from controller import Supervisor


# Контроллер робота лежит в отдельной папке, поэтому явно добавляем папку
# с общими моделями, планировщиком и правилами движения в import path.
SHARED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "warehouse_supervisor"))
# Если путь уже добавлен, второй раз его не вставляем.
if SHARED_DIR not in sys.path:
    # После этой строки робот может импортировать game_models/planner/traffic_rules.
    sys.path.insert(0, SHARED_DIR)

from game_models import Point, distance
from planner import DynamicObstacle, make_warehouse_layout
from traffic_rules import TrafficRuleEngine


# Высота центра робота над полом Webots.
ROBOT_Z = 0.14
# Высота груза, когда он лежит на складе.
CARGO_STORAGE_Z = 0.23
# Высота груза, когда он визуально лежит на платформе робота.
CARGO_CARRIED_Z = 0.47
# Линейная скорость кинематического движения робота, м/с.
ROBOT_SPEED = 0.48
# Радиус достижения waypoint: ближе этого считаем точку достигнутой.
ARRIVAL_EPS = 0.08
# Период публикации статуса робота в сеть.
STATUS_PERIOD = 0.35


@dataclass
class PeerState:
    # agent_id — id другого робота, от которого пришел status.
    agent_id: str
    # position — последняя известная позиция соседа.
    position: Point
    # battery — батарея соседа, нужна для возможных будущих политик.
    battery: float
    # status — состояние соседа: idle/to_pickup/transporting.
    status: str
    # task_id — задача соседа, если он занят.
    task_id: Optional[str]


class WarehouseRobotAgent:
    def __init__(self) -> None:
        # Используем Supervisor API локально, чтобы робот мог двигать только свой node.
        self.robot = Supervisor()
        # time_step — базовый шаг симуляции Webots в миллисекундах.
        self.time_step = int(self.robot.getBasicTimeStep())
        # getSelf возвращает Webots-узел текущего робота.
        self.self_node = self.robot.getSelf()
        # translation_field — поле, куда робот записывает собственную позицию.
        self.translation_field = self.self_node.getField("translation")
        # rotation_field — поле, куда робот записывает собственный yaw.
        self.rotation_field = self.self_node.getField("rotation")
        # Receiver принимает сообщения supervisor и других роботов.
        self.receiver = self.robot.getDevice("network_receiver")
        # Emitter отправляет status/pickup/done/replan в общую сеть.
        self.emitter = self.robot.getDevice("network_emitter")
        # Receiver должен быть включен с тем же шагом, что и контроллер.
        self.receiver.enable(self.time_step)

        # Идентификатор выводится из имени agent_0 -> R0.
        self.agent_id = self._agent_id_from_name(self.robot.getName())
        # layout содержит карту препятствий и A*.
        self.layout = make_warehouse_layout()
        # traffic_rules реализует локальные правила уступания.
        self.traffic_rules = TrafficRuleEngine(0.62)
        # position читается из текущего Webots-поля translation.
        self.position = self._read_position()
        # battery моделируется упрощенно и уменьшается от пройденного пути.
        self.battery = 1.0
        # idle означает, что робот готов принять assignment.
        self.status = "idle"
        # task_id пуст, пока робот не получил задачу.
        self.task_id: Optional[str] = None
        # route_id пуст, пока не выбран маршрут.
        self.route_id: Optional[str] = None
        # waypoints — локальный маршрут после A* уточнения.
        self.waypoints: List[Point] = []
        # waypoint_index показывает текущую цель внутри waypoints.
        self.waypoint_index = 0
        # current_task хранит полное описание назначенной задачи.
        self.current_task: Optional[dict] = None
        # tasks — последняя task board от coordinator.
        self.tasks: Dict[str, dict] = {}
        # peers — последние статусы соседних роботов.
        self.peers: Dict[str, PeerState] = {}
        # last_status_time нужен для периодической публикации, а не спама каждый tick.
        self.last_status_time = -STATUS_PERIOD
        # last_replan_time ограничивает частоту A* перепланирования.
        self.last_replan_time = -999.0
        # completed_tasks защищает от повторного принятия уже закрытой задачи.
        self.completed_tasks = set()
        # Лог старта помогает увидеть, что каждый робот запустил свой контроллер.
        print("[agent-start] %s controller ready" % self.agent_id)

    def _agent_id_from_name(self, name: str) -> str:
        # В мире роботы называются agent_0, agent_1, ...
        if name.startswith("agent_"):
            # Превращаем имя Webots в сетевой идентификатор R0/R1/...
            return "R" + name.split("_", 1)[1]
        # Если имя уже в формате R0, оставляем как есть.
        if name.startswith("R"):
            return name
        # Fallback нужен для отладочных запусков с другим именем.
        return name

    def _read_position(self) -> Point:
        # Webots хранит translation как [x, y, z].
        value = self.translation_field.getSFVec3f()
        # Для планирования нужна только плоскость X-Y.
        return (value[0], value[1])

    def run(self) -> None:
        # Главный цикл живет, пока Webots не остановит контроллер.
        while self.robot.step(self.time_step) != -1:
            # Текущее время нужно для периодов публикации.
            now = self.now()
            # Сначала обрабатываем входящие сетевые сообщения.
            self.receive_messages()
            # Затем делаем один локальный шаг движения.
            self.step_motion(self.time_step / 1000.0)
            # Записываем новую позицию робота в Webots.
            self.update_robot_node()
            # Если робот везет груз, обновляем позицию груза.
            self.update_cargo_node()
            # Публикуем status не каждый tick, а с заданным периодом.
            if now - self.last_status_time >= STATUS_PERIOD:
                self.broadcast_status()
                self.last_status_time = now

    def now(self) -> float:
        # Webots возвращает симуляционное время в секундах.
        return float(self.robot.getTime())

    def send(self, payload: dict) -> None:
        # Все сообщения получают поле from, чтобы получатель знал отправителя.
        payload["from"] = self.agent_id
        # ensure_ascii=True делает payload безопасным для Webots receiver.
        self.emitter.send(json.dumps(payload, ensure_ascii=True).encode("utf-8"))

    def receive_messages(self) -> None:
        # Receiver может накопить несколько сообщений за один tick.
        while self.receiver.getQueueLength() > 0:
            try:
                # R2023b возвращает строку через getString().
                message = json.loads(self.receiver.getString())
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Битое сообщение пропускаем, чтобы не остановить контроллер.
                self.receiver.nextPacket()
                continue
            # После чтения обязательно сдвигаем очередь Receiver.
            self.receiver.nextPacket()
            # Собственные broadcast-сообщения робот игнорирует.
            sender = message.get("from")
            if sender == self.agent_id:
                continue

            # type задает ветку протокола.
            message_type = message.get("type")
            # task_board содержит глобальное состояние задач.
            if message_type == "task_board":
                self.tasks = {item["task_id"]: item for item in message.get("tasks", [])}
            # strategy передает safety margin и другие стратегические параметры.
            elif message_type == "strategy":
                strategy = message.get("strategy", {})
                self.traffic_rules.safety_margin = float(strategy.get("safety_margin", self.traffic_rules.safety_margin))
            # assignment адресуется конкретному роботу.
            elif message_type == "assignment":
                self.accept_assignment(message)
            # status от соседей нужен для peer-to-peer collision avoidance.
            elif message_type == "status":
                self.update_peer(message)

    def update_peer(self, message: dict) -> None:
        # agent_id берется из payload, а не из поля from, чтобы протокол был явным.
        agent_id = str(message.get("agent_id", ""))
        # Пустые id и собственные сообщения игнорируем.
        if not agent_id or agent_id == self.agent_id:
            return
        # position приходит как JSON-list [x, y].
        position = message.get("position", [0.0, 0.0])
        # Обновляем или создаем запись о соседе.
        self.peers[agent_id] = PeerState(
            agent_id=agent_id,
            position=(float(position[0]), float(position[1])),
            battery=float(message.get("battery", 1.0)),
            status=str(message.get("status", "idle")),
            task_id=message.get("task_id"),
        )

    def accept_assignment(self, message: dict) -> None:
        # Робот принимает только assignment, адресованный ему.
        if message.get("agent_id") != self.agent_id:
            return
        # task содержит pickup/dropoff/deadline/cargo_def.
        task = message.get("task", {})
        # Занятый робот не принимает новую задачу; completed_tasks защищает от дублей.
        if self.status != "idle" or task.get("task_id") in self.completed_tasks:
            return

        # route содержит route_id, waypoints и edges.
        route = message.get("route", {})
        # Преобразуем JSON-list в tuple-точки для планировщика.
        route_waypoints = [tuple(point) for point in route.get("waypoints", [])]
        # Если сообщение неполное, безопасно игнорируем его.
        if not task or not route_waypoints:
            return

        # Запоминаем задачу как текущую.
        self.current_task = task
        # task_id нужен в status и правилах движения.
        self.task_id = task["task_id"]
        # route_id нужен для логов и done-сообщения.
        self.route_id = route["route_id"]
        # pickup — первая цель робота до транспортировки груза.
        pickup = tuple(task["pickup"])
        # goals включает pickup и остальные waypoints маршрута.
        goals = [pickup] + route_waypoints[1:]
        # Локально уточняем маршрут через occupancy grid/A*.
        self.waypoints = self.plan_operational_waypoints(self.position, goals)
        # Начинаем движение с первого waypoint.
        self.waypoint_index = 0
        # Меняем состояние: робот едет за грузом.
        self.status = "to_pickup"
        # Подтверждаем supervisor, что assignment принят.
        self.send({"type": "assignment_ack", "task_id": self.task_id, "route_id": self.route_id})
        # Лог нужен, чтобы видеть локальную реакцию робота.
        print("[agent-assignment] agent=%s task=%s route=%s" % (self.agent_id, self.task_id, self.route_id))

    def plan_operational_waypoints(self, start: Point, goals: Sequence[Point]) -> List[Point]:
        # cursor — текущая точка, от которой строится следующий сегмент.
        cursor = start
        # planned собирает полный локальный маршрут.
        planned: List[Point] = []
        # Каждую стратегическую цель уточняем A* отдельно.
        for goal in goals:
            segment = self.layout.plan_path(cursor, goal)
            planned.extend(segment)
            cursor = goal
        # Возвращаем плоский список waypoints.
        return planned

    def peer_obstacles(self) -> List[DynamicObstacle]:
        # Активные соседи превращаются в динамические препятствия для A*.
        return [
            DynamicObstacle(peer.agent_id, peer.position, self.layout.robot_radius)
            for peer in self.peers.values()
            if peer.status != "idle"
        ]

    def traffic_participants(self) -> List[object]:
        # Правила движения видят самого робота и всех активных соседей.
        return [self] + [peer for peer in self.peers.values() if peer.status != "idle"]

    def task_deadline(self, task_id: str) -> float:
        # Для своей текущей задачи берем deadline из current_task.
        if self.current_task and self.current_task.get("task_id") == task_id:
            return float(self.current_task.get("deadline", 999.0))
        # Для задач соседей используем последнюю task board.
        if task_id in self.tasks:
            return float(self.tasks[task_id].get("deadline", 999.0))
        # Если deadline неизвестен, ставим низкий приоритет.
        return 999.0

    def step_motion(self, dt: float) -> None:
        # Свободный робот или робот без маршрута не двигается.
        if self.status == "idle" or self.waypoint_index >= len(self.waypoints):
            return

        # target — текущий waypoint, к которому едет робот.
        target = self.waypoints[self.waypoint_index]
        # Если waypoint достигнут, переходим к следующему.
        if distance(self.position, target) < ARRIVAL_EPS:
            self.position = target
            self.waypoint_index += 1
            self.update_task_state()
            return

        # Перед движением проверяем правила уступания.
        if self.traffic_rules.must_yield(self, target, self.traffic_participants(), self.task_deadline):
            return

        # Проверяем, можно ли идти к target без статических/динамических коллизий.
        target = self.ensure_collision_free_target(target)
        # None означает, что сейчас безопасного шага нет — робот ждет.
        if target is None:
            return

        # Вектор направления движения по X.
        dx = target[0] - self.position[0]
        # Вектор направления движения по Y.
        dy = target[1] - self.position[1]
        # Длина вектора; нижняя граница защищает от деления на ноль.
        length = max(math.hypot(dx, dy), 1e-6)
        # За один tick робот не может пройти больше ROBOT_SPEED * dt.
        step = min(ROBOT_SPEED * dt, length)
        # next_position — кандидатная новая позиция.
        next_position = (self.position[0] + dx / length * step, self.position[1] + dy / length * step)
        # Финальная короткая проверка безопасности шага.
        if not self.layout.segment_is_free(self.position, next_position, dynamic=self.peer_obstacles()):
            return
        # Фиксируем новую позицию.
        self.position = next_position
        # Уменьшаем батарею пропорционально пройденному пути.
        self.battery = max(0.05, self.battery - 0.00035 * step)
        # После движения проверяем pickup/dropoff события.
        self.update_task_state()

    def ensure_collision_free_target(self, target: Point) -> Optional[Point]:
        # dynamic — текущие активные роботы-соседи.
        dynamic = self.peer_obstacles()
        # Если прямой сегмент свободен, оставляем исходный target.
        if self.layout.segment_is_free(self.position, target, dynamic=dynamic):
            return target

        # Если сегмент заблокирован, запускаем перепланирование не чаще раза в секунду.
        now = self.now()
        if now - self.last_replan_time >= 1.0:
            # A* строит локальный объезд до target.
            detour = self.layout.plan_path(self.position, target, dynamic=dynamic)
            # remaining — оставшаяся часть старого маршрута после target.
            remaining = self.waypoints[self.waypoint_index + 1 :]
            # Если A* вернул путь, подменяем текущий хвост маршрута.
            if detour:
                self.waypoints = detour + remaining
                self.waypoint_index = 0
                self.last_replan_time = now
                # Сообщаем supervisor о перепланировании для логов.
                self.send({"type": "replan", "target": list(target), "detour_len": len(detour)})

        # После перепланирования пробуем взять новый первый waypoint.
        if self.waypoint_index < len(self.waypoints):
            candidate = self.waypoints[self.waypoint_index]
            if self.layout.segment_is_free(self.position, candidate, dynamic=dynamic):
                return candidate
        # Безопасного target сейчас нет.
        return None

    def update_task_state(self) -> None:
        # Без задачи нечего обновлять.
        if not self.current_task or not self.task_id:
            return
        # pickup читаем из текущей задачи.
        pickup = tuple(self.current_task["pickup"])
        # dropoff читаем из текущей задачи.
        dropoff = tuple(self.current_task["dropoff"])

        # Если робот доехал до pickup, он начинает транспортировку.
        if self.status == "to_pickup" and distance(self.position, pickup) < ARRIVAL_EPS:
            self.status = "transporting"
            # pickup публикуется в сеть для supervisor и других наблюдателей.
            self.send({"type": "pickup", "task_id": self.task_id})
            print("[agent-pickup] agent=%s task=%s" % (self.agent_id, self.task_id))

        # Если робот с грузом доехал до dropoff, задача закрывается.
        if self.status == "transporting" and distance(self.position, dropoff) < ARRIVAL_EPS:
            # Сохраняем task_id до очистки состояния.
            finished_task = self.task_id
            # Сначала кладем груз в точку доставки.
            self.place_cargo_at_dropoff()
            # Освобождаем робота.
            self.status = "idle"
            # Помечаем задачу завершенной локально.
            self.completed_tasks.add(finished_task)
            # Отправляем done supervisor.
            self.send({"type": "done", "task_id": finished_task, "route_id": self.route_id})
            print("[agent-done] agent=%s task=%s route=%s" % (self.agent_id, finished_task, self.route_id))
            # Очищаем task_id.
            self.task_id = None
            # Очищаем route_id.
            self.route_id = None
            # Очищаем task payload.
            self.current_task = None
            # Очищаем маршрут.
            self.waypoints.clear()
            # Сбрасываем индекс waypoint.
            self.waypoint_index = 0

    def update_robot_node(self) -> None:
        # Робот сам обновляет только собственный translation.
        self.translation_field.setSFVec3f([self.position[0], self.position[1], ROBOT_Z])
        # Если есть следующая цель, разворачиваем корпус к ней.
        if self.waypoint_index < len(self.waypoints):
            target = self.waypoints[self.waypoint_index]
            dx = target[0] - self.position[0]
            dy = target[1] - self.position[1]
            if abs(dx) + abs(dy) > 1e-6:
                # Модель робота визуально смотрит вдоль +Y, поэтому yaw = atan2(-dx, dy).
                yaw = math.atan2(-dx, dy)
                self.rotation_field.setSFRotation([0, 0, 1, yaw])

    def cargo_translation_field(self):
        # Без текущей задачи робот не должен трогать груз.
        if not self.current_task or not self.task_id:
            return None
        # Находим груз по DEF-имени из task payload.
        node = self.robot.getFromDef(self.current_task["cargo_def"])
        # Если узел не найден, безопасно ничего не делаем.
        if node is None:
            return None
        # Возвращаем поле translation груза.
        return node.getField("translation")

    def place_cargo_at_dropoff(self) -> None:
        # Нужна текущая задача, чтобы знать dropoff.
        if not self.current_task:
            return
        # Получаем поле translation груза.
        field = self.cargo_translation_field()
        if field is None:
            return
        # dropoff хранится как [x, y] в task payload.
        dropoff = self.current_task["dropoff"]
        # Кладем груз на складскую высоту в точке доставки.
        field.setSFVec3f([dropoff[0], dropoff[1], CARGO_STORAGE_Z])

    def update_cargo_node(self) -> None:
        # Получаем translation груза, назначенного этому роботу.
        field = self.cargo_translation_field()
        if field is None:
            return
        # Пока робот в статусе transporting, груз следует за платформой.
        if self.status == "transporting":
            field.setSFVec3f([self.position[0], self.position[1], CARGO_CARRIED_Z])

    def broadcast_status(self) -> None:
        # Status — основное peer-to-peer сообщение для collision avoidance.
        self.send(
            {
                # Тип сообщения нужен receiver для маршрутизации.
                "type": "status",
                # agent_id дублирует from, чтобы payload был самодостаточным.
                "agent_id": self.agent_id,
                # Позицию округляем, чтобы сообщения были компактными.
                "position": [round(self.position[0], 3), round(self.position[1], 3)],
                # Батарею тоже округляем.
                "battery": round(self.battery, 4),
                # status нужен соседям для dynamic obstacle filtering.
                "status": self.status,
                # task_id нужен правилам приоритета по дедлайнам.
                "task_id": self.task_id,
                # route_id полезен для логов и анализа.
                "route_id": self.route_id,
            }
        )


if __name__ == "__main__":
    # Webots запускает файл как обычный Python-скрипт контроллера.
    WarehouseRobotAgent().run()

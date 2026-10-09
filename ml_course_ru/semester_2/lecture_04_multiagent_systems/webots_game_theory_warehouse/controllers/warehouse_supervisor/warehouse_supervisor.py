"""Network coordinator for the distributed Webots warehouse MAS example."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from controller import Supervisor

from game_models import AgentSnapshot, Assignment, Point, Task, contract_net_auction
from llm_strategy import ask_llm_strategy, fallback_strategy, strategy_to_log
from scenario import WEBOTS_ROBOT_COUNT, WEBOTS_SCENE_PROFILE, make_webots_tasks_and_routes


# DISPATCH_PERIOD задает, как часто coordinator запускает Contract Net.
DISPATCH_PERIOD = 5.0
# STRATEGY_PERIOD задает, как часто запрашивается LLM/RAG стратегия.
STRATEGY_PERIOD = float(os.environ.get("MAS_STRATEGY_PERIOD", "90.0"))
# ASYNC_LLM включает фоновый вызов Ollama, чтобы Webots-loop не останавливался.
ASYNC_LLM = os.environ.get("MAS_ASYNC_LLM", "1") != "0"
# ASYNC_LLM_TIMEOUT задает верхнюю границу фонового запроса, чтобы worker не завис навсегда.
ASYNC_LLM_TIMEOUT_ENV = os.environ.get("MAS_ASYNC_LLM_TIMEOUT", "35")
# LLM_PENDING_NOTICE_PERIOD ограничивает частоту логов о долгом ожидании LLM.
LLM_PENDING_NOTICE_PERIOD = float(os.environ.get("MAS_LLM_PENDING_NOTICE_PERIOD", "10.0"))
# BOARD_PERIOD задает частоту рассылки task board роботам.
BOARD_PERIOD = 0.7


def _parse_optional_timeout(value: str) -> Optional[float]:
    # 0/none/wait разрешают явно ждать без лимита, но для async по умолчанию стоит 35 секунд.
    normalized = value.strip().lower()
    if normalized in ("", "0", "none", "wait", "blocking", "inf", "infinite"):
        return None
    try:
        number = float(normalized)
    except ValueError:
        return 35.0
    if number <= 0.0:
        return None
    return number


ASYNC_LLM_TIMEOUT = _parse_optional_timeout(ASYNC_LLM_TIMEOUT_ENV)


def _parse_export_times(value: str) -> List[float]:
    # WEBOTS_EXPORT_IMAGES_TIMES задается как "0,120,260" в секундах симуляции.
    times: List[float] = []
    # Разбираем список вручную, чтобы не добавлять зависимостей в Webots-контроллер.
    for item in value.split(","):
        # Пропускаем пустые элементы, например после случайной завершающей запятой.
        item = item.strip()
        if not item:
            continue
        try:
            # Значения могут быть целыми или вещественными.
            times.append(float(item))
        except ValueError:
            # Ошибка в переменной окружения не должна останавливать демонстрацию.
            print("[image-export] skip invalid time=%s" % item)
    # Сортировка делает экспорт стабильным и удобным для статьи.
    return sorted(times)


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
        # Верхняя камера нужна только для экспортируемых кадров статьи.
        self.article_camera = self.robot.getDevice("article_camera")
        # Включаем приемник на каждом шаге симуляции.
        self.receiver.enable(self.time_step)
        # Камера включается всегда: без этого Camera.saveImage вернет пустой кадр.
        self.article_camera.enable(self.time_step)

        # Загружаем плотную maintenance-конфигурацию: 9 задач, 8 роботов.
        self.tasks, self.routes_by_task = make_webots_tasks_and_routes()
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
        # Стартуем мгновенно с deterministic fallback: Webots-loop не ждет Ollama.
        self.strategy = fallback_strategy(self.tasks, self.snapshots(), initial_now, WEBOTS_SCENE_PROFILE)
        # Один фоновый worker вызывает Ollama/RAG вне Webots-loop.
        self.strategy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="warehouse-llm")
        # strategy_future хранит единственный активный запрос; новые не запускаются, пока он не завершился.
        self.strategy_future: Optional[Future] = None
        # request_* нужны для логов и контроля периода.
        self.strategy_request_sim_time: Optional[float] = None
        self.strategy_request_wall_time: Optional[float] = None
        # last_pending_notice_time ограничивает частоту сообщений о долгом запросе.
        self.last_pending_notice_time = -LLM_PENDING_NOTICE_PERIOD
        # last_strategy_time равен времени запроса, поэтому следующий запрос будет через STRATEGY_PERIOD.
        self.last_strategy_time = initial_now
        # Отрицательные last_* заставляют выполнить публикацию доски и аукцион сразу после старта.
        self.last_dispatch_time = -DISPATCH_PERIOD
        self.last_board_time = -BOARD_PERIOD
        # completed_reported защищает от повторной печати success.
        self.completed_reported = False
        # WEBOTS_AUTO_QUIT=1 используется batch-тестами.
        self.auto_quit = os.environ.get("WEBOTS_AUTO_QUIT") == "1"
        # WEBOTS_EXPORT_IMAGES_DIR включает экспорт кадров сцены для статьи.
        self.export_images_dir = os.environ.get("WEBOTS_EXPORT_IMAGES_DIR", "")
        # По умолчанию сохраняем начало, середину и конец демонстрации.
        self.export_image_times = _parse_export_times(os.environ.get("WEBOTS_EXPORT_IMAGES_TIMES", "0,120,260,390"))
        # exported_images защищает от повторной записи одного и того же кадра.
        self.exported_images: set[float] = set()
        # WEBOTS_STOP_AFTER_LAST_IMAGE=1 полезен для короткой генерации иллюстраций.
        self.stop_after_last_image = os.environ.get("WEBOTS_STOP_AFTER_LAST_IMAGE") == "1"
        # Создаем папку только если пользователь явно включил экспорт.
        if self.export_images_dir:
            os.makedirs(self.export_images_dir, exist_ok=True)
            print("[image-export] dir=%s times=%s" % (self.export_images_dir, self.export_image_times))
        # Печатаем стратегию для наблюдаемости.
        print("[strategy]", strategy_to_log(self.strategy))
        # Роботы сразу получают fallback-стратегию, пока фоновая LLM готовит улучшение.
        self.broadcast_strategy()
        # Первый LLM/RAG-запрос запускается в фоне сразу после инициализации.
        self.request_strategy_update(initial_now, reason="startup")

    def _load_agents(self) -> Dict[str, NetworkAgent]:
        # agents — словарь сетевых состояний роботов.
        agents: Dict[str, NetworkAgent] = {}
        # В мире ожидаются роботы ROBOT_0..ROBOT_7.
        for index in range(WEBOTS_ROBOT_COUNT):
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

    def task_snapshots(self) -> List[Task]:
        # LLM вызывается в отдельном потоке, поэтому передаем ему копии задач.
        return [
            Task(
                task_id=task.task_id,
                pickup=task.pickup,
                dropoff=task.dropoff,
                reward=task.reward,
                deadline=task.deadline,
                cargo_def=task.cargo_def,
                status=task.status,
                assigned_agent=task.assigned_agent,
                selected_route=task.selected_route,
                assigned_at=task.assigned_at,
                completed_at=task.completed_at,
            )
            for task in self.tasks
        ]

    def request_strategy_update(self, now: float, reason: str) -> None:
        # Если async выключен переменной окружения, оставляем старый blocking-режим для отладки.
        if not ASYNC_LLM:
            print("[strategy-sync] request reason=%s time=%.1f" % (reason, now))
            self.strategy = ask_llm_strategy(
                self.task_snapshots(),
                self.snapshots(),
                now,
                scene_profile=WEBOTS_SCENE_PROFILE,
            )
            # Даже в sync-режиме обновляем go/hold по текущему состоянию прямо перед публикацией.
            self.refresh_strategy_gate(self.strategy)
            self.last_strategy_time = now
            print("[strategy]", strategy_to_log(self.strategy))
            self.broadcast_strategy()
            return

        # Не создаем очередь запросов: устаревшие LLM-ответы для движущихся роботов вреднее, чем пауза.
        if self.strategy_future is not None and not self.strategy_future.done():
            pending_since = self.strategy_request_sim_time if self.strategy_request_sim_time is not None else now
            print("[strategy-async] skip reason=%s pending_since=%.1f" % (reason, pending_since))
            self.last_strategy_time = now
            return

        # Webots API нельзя дергать из фонового потока, поэтому все данные копируются здесь.
        task_snapshot = self.task_snapshots()
        agent_snapshot = self.snapshots()
        scene_profile = dict(WEBOTS_SCENE_PROFILE)
        # submit возвращает Future; основной цикл продолжит работать с текущей стратегией.
        self.strategy_future = self.strategy_executor.submit(
            ask_llm_strategy,
            task_snapshot,
            agent_snapshot,
            now,
            ASYNC_LLM_TIMEOUT,
            scene_profile,
        )
        self.strategy_request_sim_time = now
        self.strategy_request_wall_time = time.perf_counter()
        self.last_pending_notice_time = now
        self.last_strategy_time = now
        print(
            "[strategy-async] submitted reason=%s time=%.1f current=%s"
            % (reason, now, self.strategy.note)
        )

    def poll_strategy_update(self, now: float) -> None:
        # Нет активного запроса — нечего применять.
        if self.strategy_future is None:
            return
        # Если запрос еще выполняется, основной цикл продолжает dispatch на старой стратегии.
        if not self.strategy_future.done():
            if now - self.last_pending_notice_time >= LLM_PENDING_NOTICE_PERIOD:
                request_wall_time = (
                    self.strategy_request_wall_time
                    if self.strategy_request_wall_time is not None
                    else time.perf_counter()
                )
                request_sim_time = self.strategy_request_sim_time if self.strategy_request_sim_time is not None else now
                wall_age = time.perf_counter() - request_wall_time
                print(
                    "[strategy-async] pending sim_age=%.1f wall_age=%.1f using_last_strategy=%s"
                    % (now - request_sim_time, wall_age, self.strategy.note)
                )
                self.last_pending_notice_time = now
            return

        try:
            # result() не блокирует, потому что done() уже true.
            next_strategy = self.strategy_future.result()
        except Exception as error:  # pragma: no cover - защита демонстрации от любых фоновых сбоев.
            print("[strategy-async] failed reason=%s: %s" % (error.__class__.__name__, error))
            self.strategy_future = None
            self.strategy_request_sim_time = None
            self.strategy_request_wall_time = None
            self.last_strategy_time = now
            return

        # Применяем LLM-ответ только в основном Webots-потоке.
        request_wall_time = (
            self.strategy_request_wall_time
            if self.strategy_request_wall_time is not None
            else time.perf_counter()
        )
        wall_latency = time.perf_counter() - request_wall_time
        # Async-ответ мог считаться по старому снимку, поэтому gate пересчитывается здесь.
        self.refresh_strategy_gate(next_strategy)
        self.strategy = next_strategy
        self.strategy_future = None
        self.strategy_request_sim_time = None
        self.strategy_request_wall_time = None
        self.last_strategy_time = now
        print("[strategy-async] applied time=%.1f wall_latency=%.2f" % (now, wall_latency))
        print("[strategy]", strategy_to_log(self.strategy))
        # Рассылаем новую стратегию всем robot-agent контроллерам.
        self.broadcast_strategy()

    def refresh_strategy_gate(self, strategy) -> None:
        # active_count показывает, сколько задач уже занимают pickup/dropoff/corridor capacity.
        active_count = sum(1 for task in self.tasks if task.status not in ("open", "completed"))
        # capacity_left не дает выпустить больше роботов, чем разрешает стратегия.
        capacity_left = max(strategy.max_active_tasks - active_count, 0)
        # allowed_new — сколько новых назначений можно выдать в ближайшем аукционе.
        allowed_new = max(0, min(strategy.max_new_assignments, capacity_left))
        # Если емкости нет, очищаем go_agents: contract_net сам ничего не назначит при active cap.
        if allowed_new <= 0:
            strategy.go_agents = ()
            strategy.hold_agents = tuple(agent.agent_id for agent in self.agents.values() if agent.status == "idle")
            return
        # Свободные роботы могут получить новый контракт.
        idle_agents = [agent for agent in self.agents.values() if agent.status == "idle"]
        # Семьи уже активных задач занимают те же зоны погрузки/разгрузки и коридоры.
        active_families = {
            task.task_id.rstrip("0123456789")
            for task in self.tasks
            if task.status not in ("open", "completed")
        }
        # Открытые задачи сортируются так, чтобы сначала покрывать разные свободные семьи A/B/C.
        open_tasks = sorted(
            (
                task
                for task in self.tasks
                if task.status == "open"
                and task.task_id.rstrip("0123456789") not in active_families
            ),
            key=lambda task: (
                0 if task.task_id.rstrip("0123456789") == task.task_id else 1,
                task.deadline,
                -task.reward,
                task.task_id,
            ),
        )
        # chosen хранит актуальных роботов, которых можно выпустить прямо сейчас.
        chosen: List[str] = []
        # selected_families не дает забить одну pickup/dropoff-семью первой пачкой.
        selected_families = set()
        # Первый проход выбирает ближайших роботов для разных семейств задач.
        for task in open_tasks:
            if len(chosen) >= allowed_new:
                break
            family = task.task_id.rstrip("0123456789")
            if family in selected_families:
                continue
            candidates = [agent for agent in idle_agents if agent.agent_id not in chosen]
            if not candidates:
                break
            # Для вертикальной C-линии повторную заявку устойчивее отдавать последнему C-шаттлу.
            best_agent = self.reuse_family_shuttle(family, candidates)
            # Если подходящего shuttle-робота нет, выбираем ближайшего к pickup.
            if best_agent is None:
                best_agent = min(
                    candidates,
                    key=lambda agent: (
                        (agent.position[0] - task.pickup[0]) ** 2 + (agent.position[1] - task.pickup[1]) ** 2,
                        agent.agent_id,
                    ),
                )
            chosen.append(best_agent.agent_id)
            selected_families.add(family)
        # Второй проход добирает емкость, если разных семейств меньше, чем allowed_new.
        for task in open_tasks:
            if len(chosen) >= allowed_new:
                break
            candidates = [agent for agent in idle_agents if agent.agent_id not in chosen]
            if not candidates:
                break
            family = task.task_id.rstrip("0123456789")
            # Повторное shuttle-назначение применяем и при дозаполнении емкости.
            best_agent = self.reuse_family_shuttle(family, candidates)
            # Для остальных задач достаточно ближайшего свободного агента.
            if best_agent is None:
                best_agent = min(
                    candidates,
                    key=lambda agent: (
                        (agent.position[0] - task.pickup[0]) ** 2 + (agent.position[1] - task.pickup[1]) ** 2,
                        agent.agent_id,
                    ),
                )
            if best_agent.agent_id not in chosen:
                chosen.append(best_agent.agent_id)
        # go/hold теперь соответствуют текущему состоянию, а не старому LLM snapshot.
        strategy.go_agents = tuple(chosen)
        strategy.hold_agents = tuple(agent.agent_id for agent in idle_agents if agent.agent_id not in strategy.go_agents)

    def reuse_family_shuttle(self, family: str, candidates: List[NetworkAgent]) -> Optional[NetworkAgent]:
        # C-задачи используют узкий юг-северный поток, где стабильнее повторять того же shuttle-робота.
        if family != "C":
            return None
        # candidate_ids ускоряет проверку, свободен ли робот, завершивший предыдущую C-задачу.
        candidate_ids = {agent.agent_id for agent in candidates}
        # completed_family_tasks — уже закрытые C-задачи с временем завершения.
        completed_family_tasks = [
            task
            for task in self.tasks
            if task.task_id.rstrip("0123456789") == family
            and task.status == "completed"
            and task.completed_at is not None
            and task.assigned_agent in candidate_ids
        ]
        # Если C-задач еще не закрывали, shuttle-предпочтения нет.
        if not completed_family_tasks:
            return None
        # Берем робота, который последним освободил C dropoff.
        last_task = max(completed_family_tasks, key=lambda task: task.completed_at or -1.0)
        # Возвращаем соответствующего свободного агента из candidates.
        return next((agent for agent in candidates if agent.agent_id == last_task.assigned_agent), None)

    def run(self) -> None:
        # Главный цикл coordinator живет до остановки Webots.
        try:
            while self.robot.step(self.time_step) != -1:
                # now используется для периодических действий.
                now = self.now()
                # Сначала принимаем все входящие сообщения от роботов.
                self.receive_messages()

                # Если фоновый LLM-запрос готов, применяем результат без блокировки.
                self.poll_strategy_update(now)

                # При необходимости сохраняем кадры сцены для отчета/статьи.
                self.export_images_if_needed(now)

                # Периодически обновляем стратегию через RAG+LLM, но запрос уходит в фон.
                if now - self.last_strategy_time >= STRATEGY_PERIOD:
                    self.request_strategy_update(now, reason="periodic")

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
        finally:
            # Завершаем worker без ожидания долгого Ollama-запроса при закрытии Webots.
            self.strategy_executor.shutdown(wait=False)

    def send(self, payload: dict) -> None:
        # Все сообщения coordinator помечаются отправителем SUPERVISOR.
        payload["from"] = "SUPERVISOR"
        # JSON используется как простой сетевой протокол поверх Emitter/Receiver.
        self.emitter.send(json.dumps(payload, ensure_ascii=True).encode("utf-8"))

    def export_images_if_needed(self, now: float) -> None:
        # Если переменная окружения не задана, Webots работает как обычно.
        if not self.export_images_dir:
            return
        # Проверяем все заданные моменты времени.
        for target_time in self.export_image_times:
            # Уже сохраненный кадр не перезаписываем.
            if target_time in self.exported_images:
                continue
            # Кадр сохраняется при первом tick после указанного времени.
            if now < target_time:
                continue
            # Имя файла содержит целевой момент времени, чтобы его легко цитировать.
            filename = "webots_scene_%04d.png" % int(round(target_time))
            # Полный путь находится в папке статьи или отчета.
            path = os.path.join(self.export_images_dir, filename)
            # Supervisor.exportImage экспортирует текущий вид активной сцены Webots.
            self.article_camera.saveImage(path, 95)
            # Помечаем кадр сохраненным.
            self.exported_images.add(target_time)
            # Лог нужен для воспроизводимости экспериментальных иллюстраций.
            print("[image-export] path=%s time=%.1f" % (path, now))
        # Для статей можно остановить Webots после последнего запланированного кадра.
        if (
            self.stop_after_last_image
            and self.export_image_times
            and len(self.exported_images) == len(self.export_image_times)
        ):
            print("[image-export] completed all requested frames")
            self.robot.simulationQuit(0)

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
        # Перед каждым аукционом обновляем go/hold, потому что емкость зон меняется между LLM-вызовами.
        self.refresh_strategy_gate(self.strategy)
        # Логируем диспетчерский шлюз, чтобы было видно, кто едет, а кто ждет.
        active_count = sum(1 for task in self.tasks if task.status not in ("open", "completed"))
        idle_agents = [agent.agent_id for agent in self.agents.values() if agent.status == "idle"]
        if self.strategy.go_agents or self.strategy.hold_agents:
            print(
                "[dispatch-gate] active=%d/%d max_new=%d go=%s hold=%s idle=%s blocked=%s"
                % (
                    active_count,
                    self.strategy.max_active_tasks,
                    self.strategy.max_new_assignments,
                    list(self.strategy.go_agents),
                    list(self.strategy.hold_agents),
                    idle_agents,
                    list(self.strategy.blocked_edges),
                )
            )
        # Запускаем формальную модель Contract Net + congestion game.
        assignments = contract_net_auction(
            agents=self.snapshots(),
            tasks=self.tasks,
            routes_by_task=self.routes_by_task,
            now=now,
            strategy=self.strategy,
        )
        if not assignments and any(task.status == "open" for task in self.tasks):
            print(
                "[dispatch-wait] reason=zone_capacity_or_hold active=%d max_active=%d idle=%s hold=%s"
                % (active_count, self.strategy.max_active_tasks, idle_agents, list(self.strategy.hold_agents))
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
        # go_agents — это разрешение на ближайший batch, а не постоянная роль робота.
        self.consume_go_agent(agent.agent_id)

    def consume_go_agent(self, agent_id: str) -> None:
        # Если стратегический уровень не задавал go_agents, менять нечего.
        if not self.strategy.go_agents:
            return
        # Удаляем назначенного робота из одноразового окна допуска.
        self.strategy.go_agents = tuple(item for item in self.strategy.go_agents if item != agent_id)
        # hold_agents тоже очищаем от уже занятого робота, чтобы лог не вводил в заблуждение.
        self.strategy.hold_agents = tuple(item for item in self.strategy.hold_agents if item != agent_id)
        # Лог показывает, что последующие аукционы не будут прикованы к старому списку.
        print("[dispatch-gate-consume] agent=%s remaining_go=%s" % (agent_id, list(self.strategy.go_agents)))

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
                    "go_agents": list(self.strategy.go_agents),
                    "hold_agents": list(self.strategy.hold_agents),
                    "max_active_tasks": self.strategy.max_active_tasks,
                    "max_new_assignments": self.strategy.max_new_assignments,
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

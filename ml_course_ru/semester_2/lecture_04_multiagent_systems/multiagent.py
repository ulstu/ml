import pygame  # Импортируем библиотеку Pygame для визуализации и работы с графикой.
import random  # Импортируем модуль random для генерации случайных чисел, например, для позиции агентов и задач.
import math  # Импортируем модуль math для математических операций, таких как вычисление расстояний.
from collections import defaultdict  # Импортируем defaultdict для удобного хранения данных, таких как полезность задач для агентов.

# Константы
WINDOW_SIZE = 800  # Размер окна симуляции в пикселях.
GRID_SIZE = 20  # Размер сетки, используемой для деления пространства на ячейки.
CELL_SIZE = WINDOW_SIZE // GRID_SIZE  # Размер каждой ячейки сетки, рассчитывается как размер окна, делённый на количество ячеек по одной стороне.
AGENT_COUNT = 10  # Количество агентов в симуляции.
TASK_COUNT = 20  # Количество задач в симуляции.
AGENT_SPEED = 20  # Скорость движения агентов.
MIN_AGENTS_FOR_TASK = 3  # Минимальное количество агентов, необходимое для выполнения одной задачи.
FPS = 60  # Количество кадров в секунду для управления скоростью обновления симуляции.

# Цвета
WHITE = (255, 255, 255)  # Цветовая константа для белого цвета (фон экрана).
BLACK = (0, 0, 0)  # Цветовая константа для черного цвета (линии и текст).
BLUE = (0, 0, 255)  # Цветовая константа для синего цвета (агенты).
GREEN = (0, 255, 0)  # Цветовая константа для зеленого цвета (завершенные задачи).
RED = (255, 0, 0)  # Цветовая константа для красного цвета (задачи в процессе выполнения).
GRAY = (200, 200, 200)  # Цветовая константа для серого цвета (пунктирные линии маршрута).

def distance(pos1, pos2):  # Функция для вычисления евклидового расстояния между двумя точками (pos1 и pos2).
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def normalize(vector):  # Функция для нормализации вектора, чтобы получить единичный вектор с длиной 1.
    magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
    if magnitude == 0:
        return (0, 0)
    return (vector[0] / magnitude, vector[1] / magnitude)

# Класс задачи
class Task:  # Класс Task представляет задачу, содержащую начальные и конечные точки, а также количество агентов, необходимых для выполнения.
    def __init__(self, pickup, dropoff, required_agents):  # Инициализация задачи с указанием координат начальной и конечной точки, а также требуемого количества агентов.
        self.manager = None  # Менеджер, который управляет задачей
        self.wait_cycles = 0  # Количество циклов ожидания агентов
        self.pickup = pickup
        self.dropoff = dropoff
        self.picked = False  # Груз забран
        self.completed = False  # Груз доставлен
        self.agents_assigned = []  # Список агентов, участвующих в выполнении задачи
        self.required_agents = required_agents

# Класс агента
class Agent:  # Класс Agent представляет агента, который может брать задачи, перемещаться и координироваться с другими агентами.
    def __init__(self, x, y, num=0):  # Инициализация агента с указанием начальных координат и уникального идентификатора.
        self.commitment_time = 180  # Время, на которое агент должен быть привязан к задаче перед переназначением
        self.is_manager = False  # Является ли агент менеджером задачи
        self.reassignment_cooldown = 0  # Время ожидания перед переназначением
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.task = None  # Текущая задача
        self.state = "idle"  # Состояние: idle, moving_to_pickup, moving_to_dropoff
        self.num = num

    def assign_task(self, task):  # Метод для назначения задачи агенту. Агент переходит в состояние движения к точке задачи.
        """Назначить задачу агенту"""
        if self.is_manager:
            task.manager = self
        self.task = task
        self.state = "moving_to_pickup"
        if self not in self.task.agents_assigned:
            self.task.agents_assigned.append(self)

    def move_towards(self, target):  # Метод для перемещения агента в направлении цели (target).
        """Движение к цели"""
        dx, dy = target[0] - self.x, target[1] - self.y
        direction = normalize((dx, dy))
        self.vx, self.vy = direction[0] * AGENT_SPEED, direction[1] * AGENT_SPEED
        self.x += self.vx
        self.y += self.vy
        # Если агент достиг цели
        if distance((self.x, self.y), target) < AGENT_SPEED:
            self.x, self.y = target  # Точная позиция

    def act(self, tasks, agents):  # Основная логика поведения агента, включающая назначение задач и выполнение действий в зависимости от состояния.
        """Основная логика агента"""
        if self.is_manager and self.state == "idle":  # Если агент является менеджером и находится в состоянии idle, он пытается найти других агентов для выполнения задачи.
            # Менеджер ищет исполнителей для своих задач
            if self.task and not self.task.picked and len(self.task.agents_assigned) < self.task.required_agents:
                for agent in agents:
                    if agent.state == "idle" and agent != self:
                        agent.assign_task(self.task)
                        agent.state = "moving_to_pickup"
                        if len(self.task.agents_assigned) >= self.task.required_agents:
                            break

        if self.reassignment_cooldown > 0:
            self.reassignment_cooldown -= 1
        if self.reassignment_cooldown > 0:
            self.reassignment_cooldown -= 1
        print(f'agent {self.num} acts {self.state}')
        if self.state == "idle" and self.reassignment_cooldown == 0 and not self.is_manager:
            # Проверяем, если агент должен быть привязан к текущей задаче
            if self.task and self.commitment_time > 0:
                self.commitment_time -= 1
                return
            # Агент сообщает другим агентам о наличии задачи, если сам не может её взять
            available_tasks = [t for t in tasks if not t.completed and len(t.agents_assigned) < t.required_agents]
            if available_tasks:
                for task in available_tasks:
                    for agent in agents:
                        if agent.state == "idle" and agent != self and len(task.agents_assigned) < task.required_agents:
                            agent.assign_task(task)
                            agent.state = "moving_to_pickup"
                            if len(task.agents_assigned) >= task.required_agents:
                                break
            print(f'agent {self.num} acts idle')
            # Используем формальную модель договорной сети для выбора задачи с формальной моделью ожидания
            available_tasks = [t for t in tasks if not t.completed and len(t.agents_assigned) < t.required_agents] # поиск доступных задач
            if available_tasks:
                # расчет полезности задачи
                utility = defaultdict(float)
                for task in available_tasks:
                    task_wait_penalty = task.wait_cycles / 100  # Чем дольше задача ждет, тем выше её приоритет
                    utility[task] = (1 / (1 + distance((self.x, self.y), task.pickup))) + task_wait_penalty  # Приближение по расстоянию с учетом ожидания
                # выбор лучшей задачи
                top_tasks = sorted(utility, key=utility.get, reverse=True)[:3]
                best_task = random.choice(top_tasks)
                # назначение задачи и установка параметров
                best_task.manager = self
                self.is_manager = True
                print(f'agent {self.num} {len(available_tasks)} {best_task}')
                self.assign_task(best_task)
                self.reassignment_cooldown = 60
                self.commitment_time = 180  # Устанавливаем время обязательства перед задачей  # Задержка перед возможностью переназначения
        elif self.state == "moving_to_pickup":  # Если агент движется к начальной точке задачи, проверяется, достиг ли он точки и собрались ли остальные агенты.
            print(f'agent {self.num} acts moving_to_pickup')
            if self.task and not self.task.picked:
                self.move_towards(self.task.pickup)
                if (self.x, self.y) == self.task.pickup:
                    # Проверяем, все ли агенты собрались
                    all_at_pickup = all(agent.x == self.task.pickup[0] and agent.y == self.task.pickup[1] for agent in self.task.agents_assigned)
                    # Если достаточно агентов и все на месте, начинаем доставку
                    all_at_pickup = all(agent.x == self.task.pickup[0] and agent.y == self.task.pickup[1] for agent in self.task.agents_assigned)
                    # Если достаточно агентов и все на месте, начинаем доставку
                    if len(self.task.agents_assigned) >= self.task.required_agents and all_at_pickup:
                        # Назначаем агента менеджером только когда все агенты собрались
                        self.is_manager = True
                        print(f'Task {self.task} is now being picked up by agents {[a.num for a in self.task.agents_assigned]}')
                        self.task.picked = True
                        for agent in self.task.agents_assigned:
                            agent.state = "moving_to_dropoff"
                    else:
                        # Если не хватает агентов, увеличиваем счетчик ожидания
                        self.task.wait_cycles += 1
                        print(f'wait_cycles: {self.num} {self.task.wait_cycles}')
                        if self.task.wait_cycles > 180:  # Таймаут ожидания увеличен до примерно 3 секунд при 60 FPS для более стабильного распределения
                            print(f'removed agent {self.num}')
                            self.task.agents_assigned.remove(self)
                            self.task = None
                            self.state = "idle"
                        else:
                            self.state = "waiting_for_more_agents"
        elif self.state == "moving_to_dropoff":  # Если агент движется к конечной точке задачи, проверяется, завершил ли он задачу.
            print(f'agent {self.num} acts moving_to_dropoff')
            if self.task and not self.task.completed:
                self.move_towards(self.task.dropoff)
                # Если агент - менеджер и доставили груз, освободить статус менеджера
                if (self.x, self.y) == self.task.dropoff and self.is_manager:
                    self.is_manager = False
                if (self.x, self.y) == self.task.dropoff:
                    # Завершение задачи
                    self.task.completed = True
                    if self.task:
                        for agent in self.task.agents_assigned:
                            agent.task = None
                            agent.state = "idle"
                    if self.task and self.task.agents_assigned:
                        self.task.agents_assigned.clear()  # Освобождаем агентов
                    self.task = None
                    self.state = "idle"
        elif self.state == "waiting_for_more_agents":
            # Агент сообщает другим агентам, что задача все еще нуждается в помощи
            available_tasks = [t for t in tasks if not t.completed and len(t.agents_assigned) < t.required_agents]
            if self.task in available_tasks:
                for agent in agents:
                    if agent.state == "idle" and agent != self and len(self.task.agents_assigned) < self.task.required_agents:
                        agent.assign_task(self.task)
                        agent.state = "moving_to_pickup"
                self.task.wait_cycles += 1
                print(f'wait_cycles: {self.num} {self.task.wait_cycles}')
                if self.task.wait_cycles > 120:  # Таймаут ожидания (примерно 2 секунды при 60 FPS)
                    print(f'removed agent')
                    self.task.agents_assigned.remove(self)
                    self.task = None
                    self.state = "idle"
        available_tasks = [t for t in tasks if not t.completed and len(t.agents_assigned) < t.required_agents]
        if self.task in available_tasks:
            for agent in agents:
                if agent.state == "idle" and agent != self and len(self.task.agents_assigned) < self.task.required_agents:
                    agent.assign_task(self.task)
                    agent.state = "moving_to_pickup"
            self.task.wait_cycles += 1
            print(f'wait_cycles: {self.num} {self.task.wait_cycles}')
            if self.task.wait_cycles > 120:  # Таймаут ожидания (примерно 2 секунды при 60 FPS)
                print(f'removed agent')
                self.task.agents_assigned.remove(self)
                self.task = None
                self.state = "idle"

    def draw(self, screen):  # Метод для отрисовки агента на экране. Менеджеры отображаются желтым цветом, остальные агенты — синим.
        pygame.draw.circle(screen, (255, 255, 0) if self.is_manager else BLUE, (int(self.x), int(self.y)), 8)
        if self.task:
            pygame.draw.line(screen, BLACK, (int(self.x), int(self.y)), (int(self.task.dropoff[0]), int(self.task.dropoff[1])), 1)

        elif self.state == "waiting_for_more_agents":
            if self.task and len(self.task.agents_assigned) >= self.task.required_agents:
                self.state = "moving_to_dropoff"

# Основная программа
def draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=5):  # Функция для рисования пунктирной линии между двумя точками, используется для отображения маршрута задачи.
    """Рисует пунктирную линию между двумя точками."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = dash_length
    distance = math.hypot(x2 - x1, y2 - y1)
    dash_count = int(distance / dl)
    for i in range(dash_count):
        start_x = x1 + (x2 - x1) * i / dash_count
        start_y = y1 + (y2 - y1) * i / dash_count
        end_x = x1 + (x2 - x1) * (i + 0.5) / dash_count
        end_y = y1 + (y2 - y1) * (i + 0.5) / dash_count
        pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)

def main():  # Основная функция программы, инициализирует симуляцию, создает агентов и задачи, управляет игровым циклом.
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Самоорганизующаяся сеть: Сложная транспортная задача")
    clock = pygame.time.Clock()

    # Создаём задачи
    tasks = [Task(pickup=(random.randint(0, GRID_SIZE - 1) * CELL_SIZE,
                        random.randint(0, GRID_SIZE - 1) * CELL_SIZE),
                dropoff=(random.randint(0, GRID_SIZE - 1) * CELL_SIZE,
                            random.randint(0, GRID_SIZE - 1) * CELL_SIZE),
                required_agents=random.randint(1, 5))
            for _ in range(TASK_COUNT)]

    # Создаём агентов
    agents = [Agent(random.randint(0, GRID_SIZE - 1) * CELL_SIZE,
                    random.randint(0, GRID_SIZE - 1) * CELL_SIZE, num=i) for i in range(AGENT_COUNT)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Обновляем состояние агентов
        for agent in agents:
            agent.act(tasks, agents)

        # Отрисовка
        screen.fill(WHITE)  # Заполнение экрана белым цветом перед отрисовкой новых объектов.
        # Рисуем задачи
        for task in tasks:
            color = GREEN if task.completed else (RED if task.picked else BLACK)
            pygame.draw.polygon(screen, color, [(task.dropoff[0], task.dropoff[1] - 10), (task.dropoff[0] - 10, task.dropoff[1] + 10), (task.dropoff[0] + 10, task.dropoff[1] + 10)])  # Отрисовка конечной точки задачи в виде треугольника для визуального отличия от начальной точки.
            pygame.draw.circle(screen, color, task.pickup, 5)  # Отрисовка начальной точки задачи в виде круга.
            draw_dashed_line(screen, GRAY, task.pickup, task.dropoff, 1)
            pygame.draw.polygon(screen, color, [(task.dropoff[0], task.dropoff[1] - 10), (task.dropoff[0] - 10, task.dropoff[1] + 10), (task.dropoff[0] + 10, task.dropoff[1] + 10)])
            # Рисуем информацию о необходимом количестве агентов
            font = pygame.font.Font(None, 24)  # Создание объекта шрифта для отображения информации о количестве агентов, назначенных на задачу.
            text = font.render(f"{len(task.agents_assigned)}/{task.required_agents}", True, BLACK)  # Отображение количества агентов, назначенных на задачу, и необходимого количества.
            screen.blit(text, (task.pickup[0] + 10, task.pickup[1] - 10))

        # Рисуем агентов
        for agent in agents:
            agent.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()

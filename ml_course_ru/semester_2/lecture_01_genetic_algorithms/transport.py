import random
import math
import matplotlib.pyplot as plt

# ===== Параметры задачи =====
NUM_CITIES = 40
SEED = 42
random.seed(SEED)

# Сгенерируем координаты городов (можете подставить реальные)
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(NUM_CITIES)]

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# Расчёт матрицы расстояний
D = [[dist(cities[i], cities[j]) for j in range(NUM_CITIES)] for i in range(NUM_CITIES)]

def tour_length(tour):
    s = 0.0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i+1) % n]
        s += D[a][b]
    return s

# ===== Параметры ГА =====
POP_SIZE = 300
P_CROSS = 0.9
P_MUT_BASE = 0.3          # базовая вероятность мутации (swap)
ELITE = 8                  # сколько лучших переносим без изменений
TOUR_T = 3                 # размер турнира
MAX_GEN = 400

# Иммиграция (поддержание разнообразия)
IMMIGRANTS_FRAC = 0.08     # доля худших, которых заменяем иммигрантами
IMMIGRATION_PERIOD = 30    # период (в поколениях)

# Адаптивная мутация при застое
STALL_GEN = 30             # если нет улучшений стольких поколений...
P_MUT_BOOST = 0.6          # повышаем мутацию
BOOST_LEN = 20             # на сколько поколений

# 2-opt локальный поиск (меметика)
APPLY_2OPT_TO_ELITE = True
APPLY_2OPT_TO_OFFSPRING_FRAC = 0.2  # доля потомков, к которым применим 2-opt
MAX_2OPT_PASSES = None              # None = до локального оптимума; или поставьте число итераций

# ===== Вспомогательные =====
def make_individual():
    perm = list(range(NUM_CITIES))
    random.shuffle(perm)
    return perm

def init_population():
    return [make_individual() for _ in range(POP_SIZE)]

def fitness(ind):
    # максимизируем приспособленность → берем со знаком минус длину маршрута
    return -tour_length(ind)

def tournament_selection(pop, k=TOUR_T):
    # ВАЖНО: т.к. fitness = -length, ЛУЧШЕ тот, у кого fitness БОЛЬШЕ
    best = None
    for _ in range(k):
        cand = random.choice(pop)
        if best is None or fitness(cand) > fitness(best):
            best = cand
    return best

def ox_crossover(p1, p2):
    # Ordered Crossover (OX)
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    # копируем сегмент из p1
    child[a:b+1] = p1[a:b+1]
    # заполняем свободные позиции порядком появления в p2
    p2_seq = [g for g in p2 if g not in child]
    j = 0
    for i in range(n):
        if child[i] is None:
            child[i] = p2_seq[j]
            j += 1
    return child

def swap_mutation(ind):
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]

def two_opt(route):
    # Классический 2-opt; возвращает улучшенный маршрут
    n = len(route)
    improved = True
    passes = 0
    while improved:
        improved = False
        passes += 1
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                a, b = route[i-1], route[i]
                c, d = route[j-1], route[j % n]
                if D[a][c] + D[b][d] < D[a][b] + D[c][d]:
                    route[i:j] = reversed(route[i:j])
                    improved = True
        if MAX_2OPT_PASSES is not None and passes >= MAX_2OPT_PASSES:
            break
    return route

def apply_two_opt_fraction(pop, frac):
    k = max(1, int(len(pop) * frac))
    idxs = random.sample(range(len(pop)), k)
    for i in idxs:
        pop[i] = two_opt(pop[i][:])

# ===== Основной цикл ГА =====
population = init_population()

best_per_gen = []
avg_per_gen = []
best_ever = None
best_ever_len = float('inf')

no_improve = 0
boost_left = 0

for gen in range(1, MAX_GEN + 1):
    # Оценка
    lengths = [tour_length(ind) for ind in population]
    best_idx = min(range(POP_SIZE), key=lambda i: lengths[i])
    best = population[best_idx]
    best_len = lengths[best_idx]
    avg_len = sum(lengths) / POP_SIZE

    # Статистика
    best_per_gen.append(best_len)
    avg_per_gen.append(avg_len)

    if best_len + 1e-9 < best_ever_len:
        best_ever_len = best_len
        best_ever = best[:]
        no_improve = 0
        boost_left = 0  # сбрасываем буст, если стали улучшаться
    else:
        no_improve += 1
        if no_improve >= STALL_GEN and boost_left == 0:
            boost_left = BOOST_LEN  # запустить режим повышенной мутации

    # Принты
    if gen % 20 == 0 or gen == 1:
        print(f"Поколение {gen:4d}: лучшая длина = {best_len:.2f}, средняя = {avg_len:.2f} (лучшее за всё время: {best_ever_len:.2f})")

    # Элитизм + локальное улучшение элиты
    elite_ids = sorted(range(POP_SIZE), key=lambda i: lengths[i])[:ELITE]
    elites = [population[i][:] for i in elite_ids]
    if APPLY_2OPT_TO_ELITE:
        elites = [two_opt(e[:]) for e in elites]

    # Создаем новое поколение
    new_pop = elites[:]

    # Текущая мутация (адаптивно)
    p_mut = P_MUT_BOOST if boost_left > 0 else P_MUT_BASE

    while len(new_pop) < POP_SIZE:
        # Отбор родителей
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)
        # Кроссовер
        if random.random() < P_CROSS:
            c1 = ox_crossover(p1, p2)
            c2 = ox_crossover(p2, p1)
        else:
            c1, c2 = p1[:], p2[:]
        # Мутация
        if random.random() < p_mut:
            swap_mutation(c1)
        if random.random() < p_mut:
            swap_mutation(c2)
        new_pop.extend([c1, c2])

    population = new_pop[:POP_SIZE]

    # Применим 2-opt к части потомков (поддержка «меметичности»)
    if APPLY_2OPT_TO_OFFSPRING_FRAC > 0:
        apply_two_opt_fraction(population[ELITE:], APPLY_2OPT_TO_OFFSPRING_FRAC)

    # Иммиграция по расписанию (заменяем худших на случайных)
    if IMMIGRANTS_FRAC > 0 and gen % IMMIGRATION_PERIOD == 0:
        lengths = [tour_length(ind) for ind in population]
        worst_k = max(1, int(POP_SIZE * IMMIGRANTS_FRAC))
        worst_ids = sorted(range(POP_SIZE), key=lambda i: lengths[i], reverse=True)[:worst_k]
        for wid in worst_ids:
            population[wid] = make_individual()

    # Тикаем буст
    if boost_left > 0:
        boost_left -= 1

# Финальная оценка
lengths = [tour_length(ind) for ind in population]
best_idx = min(range(POP_SIZE), key=lambda i: lengths[i])
best = population[best_idx]
best_len = lengths[best_idx]

print(f"\nЛУЧШИЙ МАРШРУТ В ФИНАЛЬНОЙ ПОПУЛЯЦИИ: длина = {best_len:.2f}")
print(f"ЛУЧШИЙ МАРШРУТ ЗА ВСЁ ВРЕМЯ:         длина = {best_ever_len:.2f}")

# ===== Визуализация =====
plt.figure()
plt.plot(best_per_gen, label="Лучшая длина")
plt.plot(avg_per_gen, label="Средняя длина")
plt.xlabel("Поколение")
plt.ylabel("Длина маршрута")
plt.title("GA для TSP: динамика качества (с 2-opt, иммигрантами, адаптивной мутацией)")
plt.legend()
plt.show()

# Рисуем лучший маршрут (за всё время)
route = best_ever
bx = [cities[i][0] for i in route] + [cities[route[0]][0]]
by = [cities[i][1] for i in route] + [cities[route[0]][1]]
plt.figure()
plt.scatter([c[0] for c in cities], [c[1] for c in cities])
plt.plot(bx, by)
plt.title(f"Лучший найденный маршрут (длина = {best_ever_len:.2f})")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

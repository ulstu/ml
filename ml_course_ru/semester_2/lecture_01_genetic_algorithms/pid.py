import random
import math
import matplotlib.pyplot as plt

# ====== Модель: m*x'' + b*x' + k*x = u ======
m, b, k = 1.0, 1.2, 20.0
U_MIN, U_MAX = -10.0, 10.0

dt = 0.002
T  = 3.0
Nsteps  = int(T / dt)

SETPOINT = 1.0

# Возмущение по управлению — чтобы ценился D-канал
DISTURB_AT  = 1.5
DISTURB_TAU = 0.10
DISTURB_U   = -4.0

# ====== PID с фильтром производной и антивиндапом ======
class PID:
    def __init__(self, Kp, Ki, Kd, N=20.0, umin=U_MIN, umax=U_MAX):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.N = N                # коэффициент фильтра производной (чем больше, тем «злее»)
        self.umin, self.umax = umin, umax
        self.reset()

    def reset(self):
        self.i = 0.0
        self.prev_y = 0.0
        self.d_state = 0.0  # состояние фильтра производной

    def step(self, r, y, dt):
        e = r - y

        # "сырой" сигнал производной по измерению
        raw_d = -(y - self.prev_y) / dt
        # 1-полюсный фильтр производной: d_state' = N * (raw_d - d_state)
        self.d_state += (self.N * (raw_d - self.d_state)) * dt

        u_unsat = self.Kp*e + self.Ki*self.i + self.Kd*self.d_state
        u = max(self.umin, min(self.umax, u_unsat))

        # Антивиндап: интегрируем только если нет сильного упора в насыщение
        if abs(u - u_unsat) < 1e-9 or (u < self.umax and u_unsat < self.umax) or (u > self.umin and u_unsat > self.umin):
            self.i += e * dt

        self.prev_y = y
        return u, e

# ====== Симулятор ======
def simulate(Kp, Ki, Kd, Ndrv, noisy=False):
    pid = PID(Kp, Ki, Kd, N=Ndrv)
    x, v = 0.0, 0.0
    y = x

    t_vals, y_vals, u_vals, e_vals = [], [], [], []
    for n in range(Nsteps):
        t = n * dt
        r = SETPOINT

        u, e = pid.step(r, y, dt)

        # Возмущение по у
        if DISTURB_AT <= t < DISTURB_AT + DISTURB_TAU:
            u += DISTURB_U

        # Динамика (Эйлер)
        a = (u - b*v - k*x) / m
        v += a * dt
        x += v * dt
        y = x

        if noisy:
            y += random.gauss(0, 0.002)

        t_vals.append(t)
        y_vals.append(y)
        u_vals.append(u)
        e_vals.append(e)

    return t_vals, y_vals, u_vals, e_vals

# ====== Критерий качества (ниже — лучше) ======
def performance_cost(y, u, t):
    e = [SETPOINT - yi for yi in y]
    dt_local = t[1] - t[0] if len(t) > 1 else dt

    # ITAE
    itae = sum(abs(ei) * ti * dt_local for ei, ti in zip(e, t))

    # Относительное перерегулирование (в долях SETPOINT)
    overshoot = max(0.0, max(y) - SETPOINT) / max(1e-9, abs(SETPOINT))

    # Время нарастания до 90%
    try:
        idx_rise = next(i for i, yi in enumerate(y) if yi >= 0.9 * SETPOINT)
        t_rise = t[idx_rise]
    except StopIteration:
        t_rise = T  # плохо, не достигли 90%

    # Установление: |e|<eps последние Ts_win
    eps = 0.02 * max(1.0, abs(SETPOINT))
    Ts_win = 0.2
    win = max(1, int(Ts_win / dt_local))
    settled = all(abs(ei) < eps for ei in e[-win:])
    t_settle_pen = 0.0 if settled else 1.0

    # Доля времени насыщения и «энергия» управления
    sat_frac = sum(1 for ui in u if ui <= U_MIN + 1e-9 or ui >= U_MAX - 1e-9) / len(u)
    energy = sum(ui * ui * dt_local for ui in u)

    # Жёсткие требования (настраиваются под ТЗ)
    TR_REQ = 0.50   # t_rise <= 0.5 c
    MP_REQ = 0.05   # overshoot <= 5%
    hard_pen = 0.0
    if t_rise > TR_REQ:
        hard_pen += 2.0 * (t_rise - TR_REQ)    # штраф за медленный подъем
    if not settled:
        hard_pen += 3.0                         # не установилось — серьёзный штраф
    if overshoot > MP_REQ:
        hard_pen += 4.0 * (overshoot - MP_REQ) # штраф за избыточный overshoot

    # Итоговая стоимость
    cost = (
        0.8 * itae +
        2.0 * t_rise +
        6.0 * overshoot +
        1.0 * t_settle_pen +
        0.8 * sat_frac +
        0.005 * energy +
        hard_pen
    )
    return cost

def evaluate(Kp, Ki, Kd, Ndrv):
    # усреднение по двум прогонам
    t, y, u, _ = simulate(Kp, Ki, Kd, Ndrv, noisy=False)
    c1 = performance_cost(y, u, t)
    t, y, u, _ = simulate(Kp, Ki, Kd, Ndrv, noisy=True)
    c2 = performance_cost(y, u, t)
    return 0.5 * (c1 + c2)

# ====== ГА для подбора [Kp, Ki, Kd, N] ======
random.seed(42)

BOUNDS = {
    "Kp": (0.10, 100.0),   # нижние границы > 0, чтобы не скатываться в чистый I
    "Ki": (0.00, 150.0),
    "Kd": (0.02, 10.0),
    "N":  (5.0,  50.0),    # фильтр производной
}

def clip(x, lo, hi): return max(lo, min(hi, x))

def make_ind():
    return [
        random.uniform(*BOUNDS["Kp"]),
        random.uniform(*BOUNDS["Ki"]),
        random.uniform(*BOUNDS["Kd"]),
        random.uniform(*BOUNDS["N"])
    ]

def fitness(ind):
    Kp, Ki, Kd, Ndrv = ind
    return -evaluate(Kp, Ki, Kd, Ndrv)   # максимизируем

def tournament_select(pop, k=3):
    best = None
    for _ in range(k):
        c = random.choice(pop)
        if best is None or c["fit"] > best["fit"]:
            best = c
    return best

def blx_crossover(p1, p2, alpha=0.35):
    child = []
    keys = ["Kp", "Ki", "Kd", "N"]
    for i, key in enumerate(keys):
        a, b = p1[i], p2[i]
        lo, hi = min(a, b), max(a, b)
        span = hi - lo
        lo2 = lo - alpha * span
        hi2 = hi + alpha * span
        val = random.uniform(lo2, hi2)
        val = clip(val, *BOUNDS[key])
        child.append(val)
    return child

def gaussian_mutation(ind, sigma=(3.0, 4.0, 0.3, 5.0), p=0.6):
    out = ind[:]
    for i, key in enumerate(["Kp", "Ki", "Kd", "N"]):
        if random.random() < p:
            out[i] += random.gauss(0.0, sigma[i])
            out[i] = clip(out[i], *BOUNDS[key])
    return out

# Параметры ГА
POP = 80
ELITE = 6
P_CROSS = 0.9
P_MUT = 0.7
GENS = 90

# Инициализация и первичная оценка
population = [{"x": make_ind(), "fit": None} for _ in range(POP)]
for p in population:
    p["fit"] = fitness(p["x"])

best_hist, avg_hist = [], []
best_ever = None

for g in range(1, GENS + 1):
    fits = [p["fit"] for p in population]
    avg_hist.append(sum(fits) / len(fits))
    best = max(population, key=lambda d: d["fit"])
    best_hist.append(-best["fit"])
    if best_ever is None or best["fit"] > best_ever["fit"]:
        best_ever = {"x": best["x"][:], "fit": best["fit"]}

    if g % 5 == 0 or g == 1:
        Kp, Ki, Kd, Ndrv = best["x"]
        print(f"Поколение {g:3d}: лучшая стоимость = {-best['fit']:.4f}  "
              f"(Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}, N={Ndrv:.1f})")

    # элитизм
    elites = sorted(population, key=lambda d: d["fit"], reverse=True)[:ELITE]
    children = elites[:]

    # генерация потомков
    while len(children) < POP:
        p1 = tournament_select(population, k=3)["x"]
        p2 = tournament_select(population, k=3)["x"]
        if random.random() < P_CROSS:
            c1 = blx_crossover(p1, p2, alpha=0.35)
            c2 = blx_crossover(p2, p1, alpha=0.35)
        else:
            c1, c2 = p1[:], p2[:]
        if random.random() < P_MUT:
            c1 = gaussian_mutation(c1)
        if random.random() < P_MUT:
            c2 = gaussian_mutation(c2)
        children.extend([{"x": c1, "fit": None}, {"x": c2, "fit": None}])

    population = children[:POP]
    for p in population:
        if p["fit"] is None:
            p["fit"] = fitness(p["x"])

# Результат
Kp, Ki, Kd, Ndrv = best_ever["x"]
print(f"\nЛучшие найденные параметры: Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}, N={Ndrv:.2f}, стоимость={-best_ever['fit']:.4f}")

# Симуляция с лучшими
t, y, u, e = simulate(Kp, Ki, Kd, Ndrv, noisy=False)

# Графики
plt.figure()
plt.plot(best_hist, label="Лучшая стоимость (min cost)")
plt.plot([-a for a in avg_hist], label="Средняя стоимость")
plt.xlabel("Поколение")
plt.ylabel("Стоимость (ниже лучше)")
plt.title("Эволюция качества PID (ГА, с N-фильтром производной)")
plt.legend()
plt.show()

plt.figure()
plt.plot(t, y, label="y(t)")
plt.axhline(SETPOINT, linestyle="--", label="задание")
plt.xlabel("t, c")
plt.ylabel("y")
plt.title(f"Отклик с PID (Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}, N={Ndrv:.1f})")
plt.legend()
plt.show()

plt.figure()
plt.plot(t, u, label="u(t)")
plt.axhline(U_MAX, linestyle="--")
plt.axhline(U_MIN, linestyle="--")
plt.xlabel("t, c")
plt.ylabel("u")
plt.title("Управляющее воздействие (с насыщением)")
plt.legend()
plt.show()

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# ================================
# ШАГ 1. СОЗДАНИЕ ПЕРЕМЕННЫХ
# ================================

# Создаем входную переменную "error" (ошибка температуры)
# Диапазон от -15 до 15 градусов с шагом 1
error = ctrl.Antecedent(np.arange(-15, 16, 1), 'error')

# Создаем входную переменную "temp_rate" (скорость изменения температуры)
# Диапазон от -5 до 5 градусов/мин
temp_rate = ctrl.Antecedent(np.arange(-5, 6, 1), 'temp_rate')

# Создаем выходную переменную "cooling_power" (мощность кондиционера)
# Диапазон от 0 до 100 процентов
cooling_power = ctrl.Consequent(np.arange(0, 101, 1), 'cooling_power')


# ================================
# ШАГ 2. ФУНКЦИИ ПРИНАДЛЕЖНОСТИ
# ================================

# --- Функции принадлежности для ошибки ---

# Очень холодно (температура намного ниже желаемой) - трапециевидная функция с пиками на -15 и -8 градусах
error['too_cold'] = fuzz.trapmf(error.universe, [-15, -15, -8, -2])

# Норма (температура близка к желаемой) - треугольная функция с пиком на 0 градусах
error['ok'] = fuzz.trimf(error.universe, [-3, 0, 3])

# Слишком жарко (температура выше желаемой) - трапециевидная функция с пиками на 2 и 8 градусах
error['too_hot'] = fuzz.trapmf(error.universe, [2, 8, 15, 15])


# --- Функции принадлежности для скорости изменения температуры ---

# Температура падает - треугольная функция с пиком на -3 градусах/мин
temp_rate['decreasing'] = fuzz.trimf(temp_rate.universe, [-5, -3, 0])

# Температура стабильна - треугольная функция с пиком на 0 градусах/мин
temp_rate['steady'] = fuzz.trimf(temp_rate.universe, [-1, 0, 1])

# Температура растет - треугольная функция с пиком на 3 градусах/мин
temp_rate['increasing'] = fuzz.trimf(temp_rate.universe, [0, 3, 5])


# --- Функции принадлежности для мощности кондиционера ---

# Низкая мощность - треугольная функция с пиком на 20 процентах
cooling_power['low'] = fuzz.trimf(cooling_power.universe, [0, 20, 40])

# Средняя мощность - треугольная функция с пиком на 50 процентах
cooling_power['medium'] = fuzz.trimf(cooling_power.universe, [30, 50, 70])

# Высокая мощность - треугольная функция с пиком на 80 процентах
cooling_power['high'] = fuzz.trimf(cooling_power.universe, [60, 80, 100])


# ================================
# ШАГ 3. НЕЧЕТКИЕ ПРАВИЛА
# ================================

# Если слишком холодно → мощность низкая
rule1 = ctrl.Rule(error['too_cold'], cooling_power['low'])

# Если температура нормальная и растет → средняя мощность
rule2 = ctrl.Rule(error['ok'] & temp_rate['increasing'], cooling_power['medium'])

# Если температура нормальная и стабильна → низкая мощность
rule3 = ctrl.Rule(error['ok'] & temp_rate['steady'], cooling_power['low'])

# Если слишком жарко и температура растет → высокая мощность
rule4 = ctrl.Rule(error['too_hot'] & temp_rate['increasing'], cooling_power['high'])

# Если слишком жарко и температура стабильна → высокая мощность
rule5 = ctrl.Rule(error['too_hot'] & temp_rate['steady'], cooling_power['high'])

# Если слишком жарко и температура падает → средняя мощность
rule6 = ctrl.Rule(error['too_hot'] & temp_rate['decreasing'], cooling_power['medium'])


# ================================
# ШАГ 4. СОЗДАНИЕ СИСТЕМЫ
# ================================

# Создаем систему управления из набора правил
cooling_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])

# Создаем объект симуляции
cooling_sim = ctrl.ControlSystemSimulation(cooling_ctrl)


# ================================
# ШАГ 5. ФУНКЦИЯ СИМУЛЯЦИИ
# ================================

def simulate_system(current_temp, desired_temp, rate):
    """
    Функция принимает:
    current_temp  - текущая температура
    desired_temp  - желаемая температура
    rate          - скорость изменения температуры
    """

    # Вычисляем ошибку регулирования
    error_value = current_temp - desired_temp

    # Передаем входные данные в систему
    cooling_sim.input['error'] = error_value
    cooling_sim.input['temp_rate'] = rate

    # Запускаем нечеткий вывод
    cooling_sim.compute()

    # Выводим результаты
    print(f"Текущая температура: {current_temp}°C")
    print(f"Желаемая температура: {desired_temp}°C")
    print(f"Ошибка: {error_value}°C")
    print(f"Скорость изменения: {rate}°C/мин")
    print(f"Мощность кондиционера: {cooling_sim.output['cooling_power']:.2f}%")

    # Визуализация результата
    cooling_power.view(sim=cooling_sim)


# ================================
# ШАГ 6. ТЕСТ СИСТЕМЫ
# ================================

# Пример:
# В комнате 30°C
# Хотим 24°C
# Температура растет на 2°C/мин
simulate_system(current_temp=25,
                desired_temp=24,
                rate=2)

# Отображаем график
plt.show()
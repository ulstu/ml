import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Шаг 1: Определение входных и выходных переменных

# Температура в помещении (°C)
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
# Желаемая температура (°C)
desired_temperature = ctrl.Antecedent(np.arange(15, 31, 1), 'desired_temperature')
# Скорость изменения температуры (°C/мин)
temp_rate = ctrl.Antecedent(np.arange(-5, 6, 1), 'temp_rate')
# Мощность кондиционера
cooling_power = ctrl.Consequent(np.arange(0, 101, 1), 'cooling_power')

# Шаг 2: Определение функций принадлежности

temperature['cold'] = fuzz.trapmf(temperature.universe, [0, 0, 10, 20])
temperature['comfortable'] = fuzz.trimf(temperature.universe, [15, 22, 28])
temperature['hot'] = fuzz.trapmf(temperature.universe, [25, 30, 40, 40])

desired_temperature['low'] = fuzz.trimf(desired_temperature.universe, [15, 17, 19])
desired_temperature['medium'] = fuzz.trimf(desired_temperature.universe, [19, 23, 27])
desired_temperature['high'] = fuzz.trimf(desired_temperature.universe, [25, 27, 30])

temp_rate['decreasing'] = fuzz.trimf(temp_rate.universe, [-5, -3, 0])
temp_rate['steady'] = fuzz.trimf(temp_rate.universe, [-1, 0, 1])
temp_rate['increasing'] = fuzz.trimf(temp_rate.universe, [0, 3, 5])

cooling_power['low'] = fuzz.trimf(cooling_power.universe, [0, 25, 50])
cooling_power['medium'] = fuzz.trimf(cooling_power.universe, [25, 50, 75])
cooling_power['high'] = fuzz.trimf(cooling_power.universe, [50, 75, 100])

# Шаг 3: Определение нечетких правил

rule1 = ctrl.Rule(temperature['cold'] & temp_rate['decreasing'], cooling_power['low'])
rule2 = ctrl.Rule(temperature['cold'] & temp_rate['steady'], cooling_power['medium'])
rule3 = ctrl.Rule(temperature['cold'] & temp_rate['increasing'], cooling_power['high'])

rule4 = ctrl.Rule(temperature['comfortable'] & temp_rate['decreasing'], cooling_power['low'])
rule5 = ctrl.Rule(temperature['comfortable'] & temp_rate['steady'], cooling_power['medium'])
rule6 = ctrl.Rule(temperature['comfortable'] & temp_rate['increasing'], cooling_power['high'])

rule7 = ctrl.Rule(temperature['hot'] & temp_rate['decreasing'], cooling_power['medium'])
rule8 = ctrl.Rule(temperature['hot'] & temp_rate['steady'], cooling_power['high'])
rule9 = ctrl.Rule(temperature['hot'] & temp_rate['increasing'], cooling_power['high'])

# Шаг 4: Создание системы управления

cooling_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
cooling_sim = ctrl.ControlSystemSimulation(cooling_ctrl)

# Функция для симуляции и визуализации

def simulate_system(temperature_value, desired_temp_value, temp_rate_value):
    cooling_sim.input['temperature'] = temperature_value
    cooling_sim.input['desired_temperature'] = desired_temp_value
    cooling_sim.input['temp_rate'] = temp_rate_value

    # Рассчитываем результат
    cooling_sim.compute()

    # Выводим результат
    print(f"Температура: {temperature_value}°C")
    print(f"Желаемая температура: {desired_temp_value}°C")
    print(f"Скорость изменения температуры: {temp_rate_value}°C/мин")
    print(f"Мощность кондиционера: {cooling_sim.output['cooling_power']:.2f}%")

    # Визуализация выходного результата
    cooling_power.view(sim=cooling_sim)

# Шаг 5: Тестирование системы

temperature_value = 30  # Температура в помещении
desired_temp_value = 25  # Желаемая температура
temp_rate_value = 2  # Скорость изменения температуры (растет)

simulate_system(temperature_value, desired_temp_value, temp_rate_value)

plt.show()

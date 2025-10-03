import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Вход 1: Ошибка по температуре (текущая - желаемая), °C
error = ctrl.Antecedent(np.arange(-20, 21, 1), 'error')
# Вход 2: Скорость изменения температуры, °C/мин
temp_rate = ctrl.Antecedent(np.arange(-5, 6, 1), 'temp_rate')
# Выход: Мощность кондиционера, %
cooling_power = ctrl.Consequent(np.arange(0, 101, 1), 'cooling_power')

# Функции принадлежности
error['too_cold']   = fuzz.trapmf(error.universe, [-20, -20, -5,  0])
error['ok']         = fuzz.trimf(error.universe,  [-2,   0,   2])
error['too_hot']    = fuzz.trapmf(error.universe, [ 0,   5,  20, 20])

temp_rate['decreasing'] = fuzz.trimf(temp_rate.universe, [-5, -3,  0])
temp_rate['steady']     = fuzz.trimf(temp_rate.universe, [-1,  0,  1])
temp_rate['increasing'] = fuzz.trimf(temp_rate.universe, [ 0,  3,  5])

cooling_power['low']    = fuzz.trimf(cooling_power.universe, [ 0, 25, 50])
cooling_power['medium'] = fuzz.trimf(cooling_power.universe, [25, 50, 75])
cooling_power['high']   = fuzz.trimf(cooling_power.universe, [50, 75,100])

# Правила (примерная логика)
rule1 = ctrl.Rule(error['too_hot'] & temp_rate['increasing'], cooling_power['high'])
rule2 = ctrl.Rule(error['too_hot'] & temp_rate['steady'],     cooling_power['high'])
rule3 = ctrl.Rule(error['too_hot'] & temp_rate['decreasing'], cooling_power['medium'])

rule4 = ctrl.Rule(error['ok'] & temp_rate['increasing'],      cooling_power['medium'])
rule5 = ctrl.Rule(error['ok'] & temp_rate['steady'],          cooling_power['low'])
rule6 = ctrl.Rule(error['ok'] & temp_rate['decreasing'],      cooling_power['low'])

rule7 = ctrl.Rule(error['too_cold'] & temp_rate['increasing'], cooling_power['low'])
rule8 = ctrl.Rule(error['too_cold'] & temp_rate['steady'],     cooling_power['low'])
rule9 = ctrl.Rule(error['too_cold'] & temp_rate['decreasing'], cooling_power['low'])

cooling_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
cooling_sim = ctrl.ControlSystemSimulation(cooling_ctrl)

def simulate_system(temperature_value, desired_temp_value, temp_rate_value):
    err = temperature_value - desired_temp_value
    cooling_sim.input['error'] = err
    cooling_sim.input['temp_rate'] = temp_rate_value
    cooling_sim.compute()

    print(f"Температура: {temperature_value}°C")
    print(f"Желаемая температура: {desired_temp_value}°C")
    print(f"Ошибка: {err:+.1f}°C")
    print(f"Скорость изменения: {temp_rate_value}°C/мин")
    print(f"Мощность кондиционера: {cooling_sim.output['cooling_power']:.2f}%")

    cooling_power.view(sim=cooling_sim)

# Тест
temperature_value = 30   # Текущая
desired_temp_value = 25  # Желаемая
temp_rate_value = 2
simulate_system(temperature_value, desired_temp_value, temp_rate_value)

plt.show()

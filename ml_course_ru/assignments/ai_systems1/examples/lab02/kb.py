import numpy as np
import matplotlib.pyplot as plt

# Треугольная функция принадлежности
def triangular_mf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)))

# Трапециевидная функция принадлежности с проверкой на деление на ноль
def trapezoidal_mf(x, a, b, c, d):
    left_slope = np.maximum(0, np.minimum(1, (x - a) / (b - a + 1e-6)))
    right_slope = np.maximum(0, np.minimum(1, (d - x) / (d - c + 1e-6)))
    return np.maximum(0, np.minimum(left_slope, right_slope))

# Универсум для температуры и влажности
x_temp = np.linspace(0, 40, 500)  # Температура от 0°C до 40°C
x_humidity = np.linspace(0, 100, 500)  # Влажность от 0% до 100%
x_cooling = np.linspace(0, 100, 500)  # Уровень охлаждения (0% до 100%)

# Определение функций принадлежности для температуры
cold = trapezoidal_mf(x_temp, 0, 0, 10, 15)
comfortable_temp = triangular_mf(x_temp, 10, 20, 25)
hot = trapezoidal_mf(x_temp, 22, 27, 40, 40)

# Определение функций принадлежности для влажности
dry = trapezoidal_mf(x_humidity, 0, 0, 30, 40)
comfortable_humidity = triangular_mf(x_humidity, 30, 50, 70)
humid = trapezoidal_mf(x_humidity, 60, 70, 100, 100)

# Определение функций принадлежности для системы охлаждения
no_cooling = trapezoidal_mf(x_cooling, 0, 0, 20, 30)
medium_cooling = triangular_mf(x_cooling, 20, 50, 80)
high_cooling = trapezoidal_mf(x_cooling, 70, 80, 100, 100)

# Правила нечеткой логики
def fuzzy_rule(temp_mf, humidity_mf):
    return np.minimum(temp_mf, humidity_mf)

# Система правил
# Система правил с добавлением нового правила
def apply_fuzzy_rules(temp_mf, humidity_mf):
    # Применение каждого правила
    rule1 = fuzzy_rule(temp_mf[0], humidity_mf[0])  # Холодно и Сухо -> Нет охлаждения
    rule2 = fuzzy_rule(temp_mf[1], humidity_mf[1])  # Комфортно и Комфортно -> Среднее охлаждение
    rule3 = fuzzy_rule(temp_mf[2], humidity_mf[2])  # Жарко и Влажно -> Высокое охлаждение
    rule4 = fuzzy_rule(temp_mf[2], humidity_mf[0])  # Жарко и Сухо -> Среднее охлаждение

    # Новое правило: Жарко и Комфортно -> Среднее охлаждение
    rule5 = fuzzy_rule(temp_mf[2], humidity_mf[1])  # Жарко и Комфортно -> Среднее охлаждение

    # Объединение выводов по всем правилам (максимум)
    cooling_output = np.maximum.reduce([np.minimum(rule1, no_cooling),
                                        np.minimum(rule2, medium_cooling),
                                        np.minimum(rule3, high_cooling),
                                        np.minimum(rule4, medium_cooling),
                                        np.minimum(rule5, medium_cooling)])  # Добавлено новое правило
    return cooling_output

# Дефаззификация (центр тяжести)
def defuzzification(cooling_output, x_cooling):
    numerator = np.sum(cooling_output * x_cooling)
    denominator = np.sum(cooling_output)
    
    if denominator == 0:
        return 0  # Если выводов нет, возвращаем 0
    return numerator / denominator

# Пример: входные данные (температура и влажность)
temp_value = 12  # Пример температуры 30°C (жарко)
humidity_value = 20  # Пример влажности 45% (сухо)

# Найдем значения функций принадлежности для заданных температуры и влажности
temp_mf = np.array([np.interp(temp_value, x_temp, cold),
                    np.interp(temp_value, x_temp, comfortable_temp),
                    np.interp(temp_value, x_temp, hot)])

humidity_mf = np.array([np.interp(humidity_value, x_humidity, dry),
                        np.interp(humidity_value, x_humidity, comfortable_humidity),
                        np.interp(humidity_value, x_humidity, humid)])

# Отладка: проверим значения функций принадлежности
print(f"Значения принадлежности для температуры {temp_value}°C: {temp_mf}")
print(f"Значения принадлежности для влажности {humidity_value}%: {humidity_mf}")

# Применяем правила
cooling_output = apply_fuzzy_rules(temp_mf, humidity_mf)

# Проверка на пустые значения после применения правил
if np.sum(cooling_output) == 0:
    print("Ошибка: все значения охлаждения равны 0.")
else:
    # Дефаззификация
    cooling_value = defuzzification(cooling_output, x_cooling)
    print(f"Для температуры {temp_value}°C и влажности {humidity_value}% уровень охлаждения: {cooling_value:.2f}%")

# Визуализация всех графиков в одном окне
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Визуализация для температуры
axs[0].plot(x_temp, cold, label="Холодно")
axs[0].plot(x_temp, comfortable_temp, label="Комфортно")
axs[0].plot(x_temp, hot, label="Жарко")
axs[0].set_title("Нечеткие множества для температуры")
axs[0].set_xlabel("Температура (°C)")
axs[0].set_ylabel("Принадлежность")
axs[0].legend()
axs[0].grid(True)

# Визуализация для влажности
axs[1].plot(x_humidity, dry, label="Сухо")
axs[1].plot(x_humidity, comfortable_humidity, label="Комфортно")
axs[1].plot(x_humidity, humid, label="Влажно")
axs[1].set_title("Нечеткие множества для влажности")
axs[1].set_xlabel("Влажность (%)")
axs[1].set_ylabel("Принадлежность")
axs[1].legend()
axs[1].grid(True)

# Визуализация системы охлаждения
axs[2].plot(x_cooling, no_cooling, label="Нет охлаждения")
axs[2].plot(x_cooling, medium_cooling, label="Среднее охлаждение")
axs[2].plot(x_cooling, high_cooling, label="Высокое охлаждение")
axs[2].set_title("Нечеткие множества для охлаждения")
axs[2].set_xlabel("Уровень охлаждения (%)")
axs[2].set_ylabel("Принадлежность")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

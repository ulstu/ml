# Задача объединения нечетких множеств с трапециевидными функциями принадлежности
import numpy as np
import matplotlib.pyplot as plt

# Функция для определения трапециевидной функции принадлежности с проверками
def trapezoidal_mf(x, a, b, c, d):
    """
    Трапециевидная функция принадлежности с проверками на деление на ноль.
    :param x: Точки, для которых вычисляется функция принадлежности.
    :param a: Левая граница начала возрастания функции.
    :param b: Левая верхняя граница (где функция равна 1).
    :param c: Правая верхняя граница (где функция равна 1).
    :param d: Правая граница окончания убывания функции.
    :return: Значение функции принадлежности в точках x.
    """
    left_slope = np.zeros_like(x)
    right_slope = np.zeros_like(x)

    # Проверяем, что a != b для корректного вычисления левой части
    if a != b:
        left_slope = (x - a) / (b - a)
    
    # Если a == b, то все значения на отрезке [a, b] равны 1
    left_slope = np.clip(left_slope, 0, 1)
    
    # Проверяем, что c != d для корректного вычисления правой части
    if c != d:
        right_slope = (d - x) / (d - c)

    # Если c == d, то все значения на отрезке [c, d] равны 1
    right_slope = np.clip(right_slope, 0, 1)
    
    return np.maximum(0, np.minimum(left_slope, np.minimum(1, right_slope)))

# Определим универсум для температуры и влажности
x_temp = np.linspace(0, 40, 500)  # Температура от 0°C до 40°C
x_humidity = np.linspace(0, 100, 500)  # Влажность от 0% до 100%

# Трапециевидные функции для температуры
cold = trapezoidal_mf(x_temp, 0, 0, 10, 15)
comfortable_temp = trapezoidal_mf(x_temp, 10, 18, 22, 25)
hot = trapezoidal_mf(x_temp, 22, 27, 35, 40)

# Трапециевидные функции для влажности
dry = trapezoidal_mf(x_humidity, 0, 0, 30, 40)
comfortable_humidity = trapezoidal_mf(x_humidity, 30, 45, 55, 70)
humid = trapezoidal_mf(x_humidity, 60, 70, 90, 100)

# Объединение двух нечетких множеств (максимум между ними)
def fuzzy_union(set1, set2):
    return np.maximum(set1, set2)

# Выполним объединение множеств для температуры
temp_union = fuzzy_union(cold, comfortable_temp)
temp_union = fuzzy_union(temp_union, hot)

# Выполним объединение множеств для влажности
humidity_union = fuzzy_union(dry, comfortable_humidity)
humidity_union = fuzzy_union(humidity_union, humid)

# Визуализация результатов
def plot_fuzzy_sets(x, sets, labels, title):
    plt.figure(figsize=(10, 6))
    for s, label in zip(sets, labels):
        plt.plot(x, s, label=label)
    plt.title(title)
    plt.xlabel("Значение")
    plt.ylabel("Принадлежность")
    plt.legend()
    plt.grid(True)
    plt.show()

# Визуализация для температуры
plot_fuzzy_sets(x_temp, [cold, comfortable_temp, hot, temp_union], 
                ["Холодно", "Комфортно", "Жарко", "Объединение"], "Нечеткие множества для температуры")

# Визуализация для влажности
plot_fuzzy_sets(x_humidity, [dry, comfortable_humidity, humid, humidity_union], 
                ["Сухо", "Комфортно", "Влажно", "Объединение"], "Нечеткие множества для влажности")

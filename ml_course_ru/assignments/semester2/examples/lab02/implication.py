# Операция импликации заданных пользователем нечетких множеств с треугольными функциями принадлежности.
import numpy as np
import matplotlib.pyplot as plt

# Треугольная функция принадлежности
def triangular_mf(x, a, b, c):
    """
    Треугольная функция принадлежности.
    :param x: Точки, для которых вычисляется функция принадлежности.
    :param a: Левая граница начала возрастания функции.
    :param b: Вершина треугольника, где принадлежность равна 1.
    :param c: Правая граница окончания убывания функции.
    :return: Значение функции принадлежности в точках x.
    """
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# Операция импликации (минимум)
def fuzzy_implication(set1, set2):
    return np.minimum(set1, set2)

# Универсум для температуры
x_temp = np.linspace(0, 40, 500)  # Температура от 0°C до 40°C

# Треугольные функции принадлежности для температуры
cold = triangular_mf(x_temp, 0, 10, 15)  # Холодно: максимум в 10°C, снижается к 15°C
comfortable_temp = triangular_mf(x_temp, 10, 20, 25)  # Комфортно: максимум в 20°C, снижается к 25°C
hot = triangular_mf(x_temp, 22, 30, 40)  # Жарко: максимум в 30°C, заканчивается в 40°C

# Выполним импликацию: "Если холодно, то комфортно"
implication_result = fuzzy_implication(cold, comfortable_temp)

# Визуализация
def plot_fuzzy_sets(x, sets, labels, title):
    plt.figure(figsize=(10, 6))
    for s, label in zip(sets, labels):
        plt.plot(x, s, label=label)
    plt.title(title)
    plt.xlabel("Температура (°C)")
    plt.ylabel("Принадлежность")
    plt.legend()
    plt.grid(True)
    plt.show()

# Визуализация нечетких множеств "Холодно", "Комфортно", "Жарко"
plot_fuzzy_sets(x_temp, [cold, comfortable_temp, hot], 
                ["Холодно", "Комфортно", "Жарко"], "Нечеткие множества для температуры")

# Визуализация результата импликации
plot_fuzzy_sets(x_temp, [implication_result], 
                ["Импликация (Если холодно -> комфортно)"], "Результат импликации")

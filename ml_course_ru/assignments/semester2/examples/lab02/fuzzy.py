import numpy as np
import matplotlib.pyplot as plt

# Функции принадлежности для температуры воздуха
def temp_low(temp):
    if temp <= 15:
        return 1
    elif 15 < temp <= 25:
        return (25 - temp) / 10
    else:
        return 0

def temp_medium(temp):
    if 15 < temp <= 25:
        return (temp - 15) / 10
    elif 25 < temp <= 35:
        return (35 - temp) / 10
    else:
        return 0

def temp_high(temp):
    if temp <= 25:
        return 0
    elif 25 < temp <= 35:
        return (temp - 25) / 10
    else:
        return 1

# Функции принадлежности для влажности почвы
def soil_dry(soil):
    if soil <= 30:
        return 1
    elif 30 < soil <= 50:
        return (50 - soil) / 20
    else:
        return 0

def soil_medium(soil):
    if 30 < soil <= 50:
        return (soil - 30) / 20
    elif 50 < soil <= 70:
        return (70 - soil) / 20
    else:
        return 0

def soil_wet(soil):
    if soil <= 50:
        return 0
    elif 50 < soil <= 70:
        return (soil - 50) / 20
    elif 70 < soil <= 90:
        return (90 - soil) / 20
    else:
        return 1

# Функции принадлежности для влажности воздуха
def air_dry(air):
    if air <= 30:
        return 1
    elif 30 < air <= 50:
        return (50 - air) / 20
    else:
        return 0

def air_medium(air):
    if 30 < air <= 50:
        return (air - 30) / 20
    elif 50 < air <= 70:
        return (70 - air) / 20
    else:
        return 0

def air_humid(air):
    if air <= 50:
        return 0
    elif 50 < air <= 70:
        return (air - 50) / 20
    else:
        return 1

# Правила нечеткой логики
def apply_rules(temp, soil, air):
    temp_levels = {
        'low': temp_low(temp),
        'medium': temp_medium(temp),
        'high': temp_high(temp)
    }
    
    soil_levels = {
        'dry': soil_dry(soil),
        'medium': soil_medium(soil),
        'wet': soil_wet(soil)
    }
    
    air_levels = {
        'dry': air_dry(air),
        'medium': air_medium(air),
        'humid': air_humid(air)
    }
    
    water_high = max(
        min(temp_levels['high'], soil_levels['dry']),
        min(temp_levels['high'], soil_levels['medium'], air_levels['dry'])
    )
    
    water_medium = max(
        min(temp_levels['medium'], soil_levels['dry']),
        min(temp_levels['high'], soil_levels['medium']),
        min(temp_levels['medium'], soil_levels['medium'], air_levels['medium']),
        min(temp_levels['low'], soil_levels['medium'], air_levels['dry'])
    )
    
    water_low = max(
        min(temp_levels['low'], soil_levels['wet']),
        min(temp_levels['medium'], soil_levels['wet'], air_levels['humid']),
        min(temp_levels['low'], soil_levels['medium'], air_levels['humid'])
    )
    
    return water_low, water_medium, water_high

# Дефаззификация методом центра тяжести (центр масс)
def defuzzification(water_low, water_medium, water_high):
    x = np.linspace(0, 100, 1000)
    
    low_membership = np.array([min(water_low, 1 - i/100) for i in x])
    medium_membership = np.array([min(water_medium, 1 - abs((i-50)/50)) for i in x])
    high_membership = np.array([min(water_high, i/100) for i in x])
    
    numerator = np.sum(x * np.maximum.reduce([low_membership, medium_membership, high_membership]))
    denominator = np.sum(np.maximum.reduce([low_membership, medium_membership, high_membership]))
    
    return numerator / denominator if denominator != 0 else 0

# Визуализация влажности почвы
def plot_results(time, soil_moisture):
    plt.figure(figsize=(10, 6))
    plt.plot(time, soil_moisture, label="Влажность почвы", color="green")
    plt.xlabel("Время (дни)")
    plt.ylabel("Влажность почвы (%)")
    plt.title("Изменение влажности почвы для сахарной свеклы")
    plt.grid(True)
    plt.legend()
    plt.show()

# Моделирование системы полива на протяжении 6 месяцев с поливом каждые 3-7 дней
def irrigation_simulation(months=6, time_step_hours=12, initial_soil_moisture=40):
    # Параметры симуляции
    time_steps = int((months * 30 * 24) / time_step_hours)  # Количество временных шагов
    soil_moisture = initial_soil_moisture  # Начальная влажность почвы
    soil_moisture_history = []
    time_history = []
    
    # Параметры почвы для сахарной свеклы
    max_soil_moisture = 70  # Максимальная влажность почвы (состояние полного насыщения)
    min_soil_moisture = 30  # Минимальная влажность почвы (стресс для сахарной свеклы)
    
    # Интервалы полива (раз в 3-7 дней)
    watering_intervals = np.random.randint(3, 8, size=(time_steps // 14)) * 2  # Раз в 3-7 дней, 12 часов на шаг
    
    next_watering_step = watering_intervals[0]
    interval_idx = 0
    dry_days = 0  # Счетчик дней без полива

    # Моделирование погодных условий и полива
    for t in range(time_steps):
        # Генерация случайных значений температуры и влажности воздуха
        temp = np.random.uniform(10, 35)  # Температура от 10°C до 35°C
        air_humidity = np.random.uniform(20, 80)  # Влажность воздуха от 20% до 80%
        
        # Проверка времени полива
        if t == next_watering_step:
            water_low, water_medium, water_high = apply_rules(temp, soil_moisture, air_humidity)
            watering_level = defuzzification(water_low, water_medium, water_high)
            soil_moisture += watering_level * 0.1  # Полив увеличивает влажность почвы (коэффициент полива)
            
            # Установить следующий полив
            interval_idx = (interval_idx + 1) % len(watering_intervals)
            next_watering_step += watering_intervals[interval_idx]
            dry_days = 0  # Сброс счетчика сухих дней
        else:
            dry_days += 1  # Увеличение счетчика сухих дней

        # Снижение влажности при отсутствии полива (медленное снижение через 2 дня)
        if dry_days >= 2:
            soil_moisture -= 1  # Уменьшение влажности при отсутствии полива (уменьшается на 1% каждые 2 дня)
        
        # Испарение и снижение влажности почвы
        soil_moisture -= 0.01 * (35 - air_humidity)  # Уменьшение влажности почвы из-за испарения
        
        # Ограничение влажности почвы
        soil_moisture = max(min_soil_moisture, min(max_soil_moisture, soil_moisture))
        
        # Сохранение истории
        soil_moisture_history.append(soil_moisture)
        time_history.append(t * time_step_hours / 24)  # Время в днях

    # Визуализация результатов
    plot_results(time_history, soil_moisture_history)

# Тестирование системы полива на протяжении 6 месяцев для сахарной свеклы
irrigation_simulation()

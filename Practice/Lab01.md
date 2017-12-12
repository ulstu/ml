# English
# Assignment for practical work 1
## General Assignment
1. Generate csv file containing 3 columns: x1, x2, y (> 400 rows). The type of the function is determined by the option (the range is chosen by the student).
2. Open the file and create 2 plots in one window: x1 (x) and x2 (x) on one plot and y (x1), y (x2) on the another. Plots need to be built with matplotlib library (for the function y, you need to display points on the plot).
3. Print to the console for each column (x1, x2, y) average, minimum and maximum values.
4. Save data to the new csv file those lines for which the condition: x1 less average_x1 or x2 less average_x2
5. Usу mplot3D to build a 3D graph of the function y (x1, x2) in a separate window.

## Options
y is a random number
1. x1 = sin(x); x2 = cos(x)
2. x1 = tg(x); x2 = sin(x)
3. x1 = x^6 + x^2 + x^3 + 4x + 5; x2 = 2 + 4x + 5x^2
4. x1 = 5 - log(x); x2 = - 1 / sqrt(x)
5. x1 = x^2 - 50; x2 = 0.001 * x
6. x1 = sin(x) * x ^ 2; x2 = -x^2 + 6
7. x1 = cos(x) * x^3; x2 = -3 * x^3 + 7
8. x1 = 5 / (1 + e ^ (-x)); x2 = x ^ 2
9. x1 = 3 / (2 + e ^ (-2x)); x2 = 0.01 * th(x) * x ^ 2 
10. x1 = 4 / (1 + e ^ (-x)); x2 = sin(x)
11. x1 = 1 / (1 + e ^ (-x)); x2 = sin(x) + 0.7
12. x1 = (2x + x^2 + 0.5 * x^3 + x^4 + 2 * x^5), x2 = 20000 * sin(x)


# Russian
# Задание на лабораторную работу №1
## Общее задание
1. Сгенерировать csv файл, содержащий 3 столбца: x1, x2, y (> 400 строк). Вид функции определяется вариантом (диапазон выбирается студентом).
2. Открыть файл и сформировать 2 графика в одном окне: x1(x) и x2(x) на одном графике и y(x1), y(x2) на другом. Графики необходимо построить с использованием matplotlib (для функции y отобразить точки на графике). 
3. Вывести на консоль для каждого столбца (x1, x2, y): среднее, минимальное и максимальное значения.
4. Сохранить в новый csv файл те строки, для которых выполняется условие: x1 меньше среднее_x1 или x2 меньше среднее_x2
5. С использованием mplot3D построить 3D график функции y(x1, x2) в отдельном окне.

## Варианты
y - произвольное число (random)

1. x1 = sin(x); x2 = cos(x)
2. x1 = tg(x); x2 = sin(x)
3. x1 = x^6 + x^2 + x^3 + 4x + 5; x2 = 2 + 4x + 5x^2
4. x1 = 5 - log(x); x2 = - 1 / sqrt(x)
5. x1 = x^2 - 50; x2 = 0.001 * x
6. x1 = sin(x) * x ^ 2; x2 = -x^2 + 6
7. x1 = cos(x) * x^3; x2 = -3 * x^3 + 7
8. x1 = 5 / (1 + e ^ (-x)); x2 = x ^ 2
9. x1 = 3 / (2 + e ^ (-2x)); x2 = 0.01 * th(x) * x ^ 2 
10. x1 = 4 / (1 + e ^ (-x)); x2 = sin(x)
11. x1 = 1 / (1 + e ^ (-x)); x2 = sin(x) + 0.7
12. x1 = (2x + x^2 + 0.5 * x^3 + x^4 + 2 * x^5), x2 = 20000 * sin(x)
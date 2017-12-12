# -*- coding: utf-8 -*-
import pandas
import pandas.core.series as Series
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# TASK 4
# How old were the passengers?
# Calculate the average and median age of passengers
mean_age = data['Age'].mean()
median_age = data['Age'].median()

# print(data['Age'])
print(mean_age)
print(median_age)
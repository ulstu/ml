# -*- coding: utf-8 -*-
import pandas
import pandas.core.series as Series
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# TASK 3
# What percentage of the first class passengers were among all passengers?

total_count = len(data)
first_class_count = len(data.loc[data['Pclass'] == 1])

first_class_persent = float(first_class_count)/total_count
print first_class_persent, "%"
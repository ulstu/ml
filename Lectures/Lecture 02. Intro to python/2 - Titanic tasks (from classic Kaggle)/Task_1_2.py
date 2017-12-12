# -*- coding: utf-8 -*-
import pandas
import pandas.core.series as Series
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# TITANIC TASK - 2
# Which part of the passengers managed to survive?

total_count = len(data)
survived_count = data.loc[data['Survived'] == 1].count().values[0]
survived_persent = float(survived_count)/total_count
print(survived_persent * 100, "%")
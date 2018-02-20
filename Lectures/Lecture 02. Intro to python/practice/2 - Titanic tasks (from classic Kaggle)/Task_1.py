# -*- coding: utf-8 -*-
import pandas
import pandas.core.series as Series
import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# TITANIC TASK - 1
# print Table headers instead all data table
print(list(data))
# print data

# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex
# Age	Age in years
# sibsp	# of siblings / spouses aboard the Titanic
# parch	# of parents / children aboard the Titanic
# ticket	Ticket number
# fare	Passenger fare
# cabin	Cabin number
# embarked	Port of Embarkation

# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
#
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
#
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
#
# parch: The dataset defines family relations in this way...
# 1Parent = mother, father
# 2Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# TASK 1
# How many men and women were traveling by ship?

# [[colName]] - return DataFrame object,
# [colName] - return Series object
# see more - http://pandas.pydata.org/pandas-docs/stable/
mans_count = len(data.loc[data['Sex'] == 'male'])
sex = data.loc[data['Sex'] == 'male']
# print(sex)
# for get matrix of digits use values[0]
womans_count = sex.loc[data['Sex'] == 'female'].count().values[0]
print(mans_count)
print(womans_count)

















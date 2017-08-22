# -*- coding: utf-8 -*-
import pandas
import pandas.core.series as Series
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# TITANIC TASK - 1
# print Table headers instead all data table
print list(data)
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
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# TASK 1
# How many men and women were traveling by ship?

# [[colName]] - двойные скобки возвращают объект DataFrame,
# их нужно использовать если выбирается несколько столбцов
# или если с одним столбцом дальше планируется взаимодействовать как с DataFrame
# если же нужно только выбрать столбец и проверить некоторое условие,
# то можно использовать одинарные скобки [colName]
# ниже представлены оба примера

# после метода count() возвращается объект pandas.core.series.Series
# поэтому для получения непосредственно значения можно использовать values[0]

mans_count = len(data.loc[data['Sex'] == 'male'])
sex = data[['Sex']]
womans_count = sex.loc[data['Sex'] == 'female'].count().values[0]
print mans_count
print womans_count

















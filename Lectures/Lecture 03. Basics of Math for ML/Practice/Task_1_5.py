# -*- coding: utf-8 -*-
import pandas
import pandas.core.series as Series
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# TASK 5
# Do the number of brothers / sisters / spouses correlate with the number of parents / children?
# Calculate the Pearson correlation between the SibSp and Patch features

corr = data[['Parch', 'SibSp']].corr()
print corr
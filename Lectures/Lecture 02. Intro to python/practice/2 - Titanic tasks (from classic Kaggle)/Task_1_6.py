# -*- coding: utf-8 -*-
import pandas
import pandas.core.series as Series
import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')


# TASK 6
# What is the most popular female name on the ship?


def get_first_name(full_name):
    first_name = 'null'
    words = full_name.split()
    index = 0
    if "Miss." in words:
        index = words.index("Miss.")
    if "Missis." in words:
        index = words.index("Missis.")
    first_name = words[index + 1]
    return first_name


# apply get_first_name function for each cell in Name column
data['Name'] = data['Name'].apply(get_first_name)
print(data['Name'])
#  Complete code for this is presented below (but it longer):
#  data['Name'] = data['Name'].apply(lambda x: get_first_name(x))

# Select rows where Name not null (because null mean not woman names)
onlyGirlsFirstNames = data.loc[data['Name'] != 'null']

# after groupby function table structure is changed, that's why we need
# reindex data frame use reset_index()
girlsNamesCount = onlyGirlsFirstNames.groupby(['Name']).size().reset_index()

# because table struct is changed and now it have new column we also need redefine columns
girlsNamesCount.columns = ['Name', 'Count']

# function sort got 2 args:
# - name or index of column as sort base
# - sort direction (ascending or descending)
girlsNamesCount_Sort = girlsNamesCount.sort_values(by=['Count'], ascending=False)
print(girlsNamesCount_Sort)

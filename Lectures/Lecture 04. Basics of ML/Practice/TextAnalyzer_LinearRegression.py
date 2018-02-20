# -*- coding: utf-8 -*-
import pandas
import pandas.core.series as Series
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

# read source data
data_train = pandas.read_csv('salary-train.csv')
data_test = pandas.read_csv('salary-test-mini.csv')

print(list(data_train))

# set lowercase text
data_train['FullDescription'] = data_train.FullDescription.str.lower()
data_test['FullDescription'] = data_test.FullDescription.str.lower()
# remove not need symbols
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

# print data_train

# map text to vector use TdifVectorizer in sklearn
vectorizer = TfidfVectorizer(min_df=10)
# RETURN CPARSE MATRIX
train_text_feature_matrix = vectorizer.fit_transform(data_train['FullDescription'])
# print train_text_feature_matrix
test_text_feature_matrix = vectorizer.transform(data_test['FullDescription'])
# idf = vectorizer.idf_
# print dict(zip(vectorizer.get_feature_names(), idf))


# fill empty cells
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

# map text categorical feature to numeric code
enc = DictVectorizer()
train_dic = data_train[['LocationNormalized', 'ContractTime']].to_dict('records')
test_dic = data_test[['LocationNormalized', 'ContractTime']].to_dict('records')
# print train_dic

X_train_categ = enc.fit_transform(train_dic)
X_test_categ = enc.transform(test_dic)

# print test_text_feature_matrix
# print X_train_categ

x_train = hstack((train_text_feature_matrix, X_train_categ))
y_train = data_train['SalaryNormalized'].values
x_test = hstack((test_text_feature_matrix, X_test_categ))

# create model of linear regression with regularization and train it
# train and test it
ridge_regression = Ridge(alpha=1, random_state=241)
ridge_regression.fit(x_train, y_train)
y_test = ridge_regression.predict(x_test)
print(y_test)

# create model of linear regression without regularization
# train and test it
ridge_regression = Ridge(alpha=0, random_state=241)
ridge_regression.fit(x_train, y_train)
y_test = ridge_regression.predict(x_test)
print(y_test)

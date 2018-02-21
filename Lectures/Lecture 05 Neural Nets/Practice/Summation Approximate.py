# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons, make_circles, make_classification, make_friedman1, make_friedman2
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



X = [[1, 1], [2, 2], [3, 3], [1, 4], [1, 10], [5, 10], [0, 1], [1, 0], [0,0]]
y = [2, 4, 6, 5, 11, 14, 1, 1, 0]

summation = X, y

datasets = [summation]

data_names = ["summation"]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    # X = StandardScaler().fit_transform(X)
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    lr = LinearRegression(normalize=True)
    lr.fit(X_train, Y_train)

    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, Y_train)

    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, Y_train)

    # rfr = RandomForestRegressor()
    # rfr.fit(X_train, Y_train)

    mlp = MLPRegressor(hidden_layer_sizes=(30,), max_iter=1000)
    mlp.fit(X_train, Y_train)

    from sklearn.metrics import accuracy_score

    acc_lr = lr.score(X_test, Y_test)
    acc_lasso = lasso.score(X_test, Y_test)
    acc_ridge = ridge.score(X_test, Y_test)
    # acc_rfr = rfr.score(X_test, Y_test)
    acc_mlp = mlp.score(X_test, Y_test)

    print("LinearRegression: ", acc_lr)
    print("Lasso: ", acc_lasso)
    print("Ridge: ", acc_ridge)
    # print "RandomForestRegressor: ", acc_rfr
    print("MLPRegressor: ", acc_mlp)


X_control = [[7, 3], [12, 3], [20, 3], [40,40], [500, 300]]
print(lr.predict(X_control))
print(mlp.predict(X_control))
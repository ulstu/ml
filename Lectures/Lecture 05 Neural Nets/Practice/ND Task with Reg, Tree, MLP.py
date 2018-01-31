# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_friedman1, make_friedman2
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron


names = ["LinearRegression",
                "polynom 4",
                "polynom 5",
                "ridge_polynom 5",
                "Perceptron",
                "MLP 10",
                "MLP 100"]

polynomic_regression_low = Pipeline([('poly', PolynomialFeatures(degree=4)),
                                 ('linear', LinearRegression(fit_intercept=False))])

polynomic_regression_high = Pipeline([('poly', PolynomialFeatures(degree=5)),
                                 ('linear', LinearRegression(fit_intercept=False))])

polynomic_ridge_high = Pipeline([('poly', PolynomialFeatures(degree=5)),
                                 ('linear', Ridge(alpha=2.0))])

classifiers = [
    LinearRegression(),
    polynomic_regression_low,
    polynomic_regression_high,
    polynomic_ridge_high,
    Perceptron(random_state=0),
    MLPClassifier(alpha=0.01, hidden_layer_sizes=(10,)),
    MLPClassifier(alpha=0.01, hidden_layer_sizes=(100,))
]

X, y = make_classification(n_samples=500, n_features=30, n_redundant=5, n_repeated=1, n_informative=24,
                           random_state=1, n_clusters_per_class=1, class_sep=0.7, shift=0.3)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [linearly_separable]

data_names = ["moons", "circles", "linearly_separable"]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print name, score


# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_friedman1, make_friedman2
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Perceptron
import pylab


h = .02  # step size in the mesh
rs = 25

# define models name
names = ["LinearRegression",
                "polynom 3",
                "polynom 4",
                "ridge_polynom 4",
                "Perceptron",
                "MLP 10",
                "MLP 100"]

# create models
polynomic_step = ('poly', PolynomialFeatures(degree=3))
linear_step = ('linear', LinearRegression(fit_intercept=False))
polynomic_regression_low = Pipeline([polynomic_step, linear_step])

polynomic_regression_high = Pipeline([('poly', PolynomialFeatures(degree=4)),
                                 ('linear', LinearRegression(fit_intercept=False))])

polynomic_ridge_high = Pipeline([('poly', PolynomialFeatures(degree=4)),
                                 ('linear', Ridge(alpha=1.0, random_state=rs))])

classifiers = [
    LinearRegression(),
    polynomic_regression_low,
    polynomic_regression_high,
    polynomic_ridge_high,
    Perceptron(random_state=rs),
    MLPClassifier(alpha=0.01, hidden_layer_sizes=(10,), random_state=rs),
    MLPClassifier(alpha=0.01, hidden_layer_sizes=(100,), random_state=rs)
]

# generate source data
# first linear separable data set
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2,
                           random_state=rs, n_clusters_per_class=1)
rng = np.random.RandomState(rs)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# and than non linear data set
# moons and circles
datasets = [
    linearly_separable,
    make_moons(noise=0.3, random_state=rs),
    make_circles(noise=0.2, factor=0.5, random_state=rs)]

data_names = ["moons", "circles", "linearly_separable"]

# prepare figure for plotting
figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=rs)

    x0_min, x0_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x1_min, x1_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    current_subplot = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        current_subplot.set_title("Input data")
    # Plot the training points
    current_subplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    current_subplot.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    current_subplot.set_xlim(xx0.min(), xx0.max())
    current_subplot.set_ylim(xx1.min(), xx1.max())
    current_subplot.set_xticks(())
    current_subplot.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        current_subplot = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx0.ravel(), xx1.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx0.ravel(), xx1.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx0.shape)
        current_subplot.contourf(xx0, xx1, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        # and testing points
        current_subplot.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,alpha=0.6)

        current_subplot.set_xlim(xx0.min(), xx0.max())
        current_subplot.set_ylim(xx1.min(), xx1.max())
        current_subplot.set_xticks(())
        current_subplot.set_yticks(())
        if ds_cnt == 0:
            current_subplot.set_title(name)
        current_subplot.text(xx0.max() - .3, xx1.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score


def MakeExample(index, plt, model):
    # create polynomic coefs (features) for polynomic regression
    polynomial_features = PolynomialFeatures(degree=degrees[index],
                                             include_bias=False)

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", model)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    # visualization source data (points)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))

    return plt


np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

# define function which we try approximate with model
true_fun = lambda X: np.cos(1.5 * np.pi * X)
X = np.sort(np.random.rand(n_samples))
# add some random offset
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 6))

# process linear regression
for i in range(len(degrees)):
    linear_regression = LinearRegression()
    plt.subplot(2, len(degrees), i+1)
    plt = MakeExample(i, plt, linear_regression)

# process ridge regression
for i in range(len(degrees)):
    ridge = Ridge(alpha=0.02)
    plt.subplot(2, len(degrees), len(degrees) + i + 1)
    plt = MakeExample(i, plt, ridge)

plt.show()

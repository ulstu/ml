import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


# data_train = pandas.read_csv('perceptron-train.csv')
# X_train = data_train.ix[:, 1:3].values
# y_train = data_train.ix[:, 0].values
#
# data_test = pandas.read_csv('perceptron-test.csv')
# X_test = data_test.ix[:, 1:3].values
# y_test = data_test.ix[:, 0].values
#
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

# X, y = make_moons(noise=0.3, random_state=0)
# X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
figure = plt.figure(figsize=(17, 9))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, 1, 1)
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
plt.show()

# alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
alphas = [0.1]
neurons = [10, 50, 100, 200, 400, 800]
L2_coefs = [0, 0.1]

result = {}

for l2_c in L2_coefs:
    for a in alphas:
        for n in neurons:
            if(l2_c==0):
                mlp = MLPClassifier(solver="adam", activation="tanh", alpha=a, hidden_layer_sizes=(n, ), max_iter=5000, tol=0.00000001)
            else:
                mlp = MLPClassifier(solver="adam", activation="tanh", alpha=a, hidden_layer_sizes=(n, l2_c*n), max_iter=5000, tol=0.00000001)
            mlp.fit(X_train, y_train)
            predictions = mlp.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            description = "L1 = ", n, "L2 = ", l2_c*n, "alpha = ", a, "Acc = "
            result[description] = acc

            # print description, acc


result = sorted(result.iteritems(), key=lambda x, y: y, reverse=True)

for key, value in result:
    print(key, value)
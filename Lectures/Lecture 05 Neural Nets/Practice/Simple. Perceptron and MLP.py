import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def median(lst):
    return np.median(np.array(lst))


# read train data from file as table
data_train = pandas.read_csv('perceptron-train.csv')
# get feature matrix and Y values from source table
X_train = data_train.ix[:, 1:3].values
y_train = data_train.ix[:, 0].values

# read test data from file as table
data_test = pandas.read_csv('perceptron-test.csv')
# get feature matrix and Y values from source table
X_test = data_test.ix[:, 1:3].values
y_test = data_test.ix[:, 0].values

acc_p = []
acc_pn = []
acc_mlp = []
acc_mlpn = []

# make a few experiments with different random state
# (with different start position in gradient space)
for i in range(10):
    print "Random: ", i
    # create and train model of perceptron
    clf = Perceptron(random_state=i, alpha=0.01, n_iter=2000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    # evaluate model of perceptron
    acc = accuracy_score(y_test, predictions)
    print "Perceptron: ", acc
    acc_p.append(acc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # create and train model of perceptron with
    # data scaling before
    clf = Perceptron(random_state=i, alpha=0.01, n_iter=2000)
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print "Perceptron with normalization: ", acc
    acc_pn.append(acc)

    # create and train model of multi layer perceptron
    mlp = MLPClassifier(random_state=i, solver="sgd", activation="tanh", alpha=0.01, hidden_layer_sizes=(2,),
                        max_iter=2000, tol=0.00000001)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print "MLP: ",  acc
    acc_mlp.append(acc)

    # create and train model of multi layer perceptron  with
    # data scaling before
    mlp = MLPClassifier(random_state=i, solver="sgd", activation="tanh", alpha=0.01, hidden_layer_sizes=(2,),
                        max_iter=2000, tol=0.00000001)
    mlp.fit(X_train_scaled, y_train)
    predictions = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print "MLP: ",  acc
    acc_mlpn.append(acc)

# comparasion models
print "Perceptron: ", min(acc_p), median(acc_p), max(acc_p), np.std(acc_p)
print "Perceptron with Norm: ", min(acc_pn), median(acc_pn), max(acc_pn), np.std(acc_pn)
print "MLP: ", min(acc_mlp), median(acc_mlp), max(acc_mlp), np.std(acc_mlp)
print "MLP with Norm: ", min(acc_mlpn), median(acc_mlpn), max(acc_mlpn), np.std(acc_mlpn)

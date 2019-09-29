from sklearn.linear_model import (LinearRegression, Ridge,
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import collections
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

np.random.seed(0)

size = 750
X_train = np.random.uniform(0, 1, (size, 14))
X_train[:,10:] = X_train[:,:4] + np.random.normal(0, .025, (size,4))

Y_train = (10 * np.sin(np.pi*X_train[:,0]*X_train[:,1]) + 20*(X_train[:,2] - .5)**2 +
     10*X_train[:,3] + 5*X_train[:,4]**5 + np.random.normal(0,1))

X_test = np.random.uniform(0, 1, (size, 14))
X_test[:,10:] = X_test[:,:4] + np.random.normal(0, .025, (size,4))

Y_test = (10 * np.sin(np.pi*X_test[:,0]*X_test[:,1]) + 20*(X_test[:,2] - .5)**2 +
     10*X_test[:,3] + 5*X_test[:,4]**5 + np.random.normal(0,1))

lr = LinearRegression(normalize=True)
lr.fit(X_train, Y_train)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, Y_train)

ridge = Ridge(alpha=0.1)
ridge.fit(X_train, Y_train)

rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)

mlp = MLPRegressor(hidden_layer_sizes=(200,), max_iter=1000)
mlp.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score

acc_lr = lr.score(X_test, Y_test)
acc_lasso = lasso.score(X_test, Y_test)
acc_ridge = ridge.score(X_test, Y_test)
acc_rfr = rfr.score(X_test, Y_test)
acc_mlp = mlp.score(X_test, Y_test)

print("LinearRegression: ",acc_lr)
print("Lasso: ",acc_lasso)
print("Ridge: ",acc_ridge)
print("RandomForestRegressor: ", acc_rfr)
print("MLPRegressor: ",acc_mlp)
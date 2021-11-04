# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import linear_model

# set normal distribution
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)

#  define synthetic train function which we try approximate with model
y = (X > 0).astype(np.float)
# rescaling by X
X[X > 0] *= 4
# add random offset by X
X += .3 * np.random.normal(size=n_samples)

# pack 1-dimension array to 2-d
X = X[:, np.newaxis]

# generate regular points for test
X_test = np.linspace(start=-5, stop=10, num=300)
X_test = X_test[:, np.newaxis]

# define function for test same as train
y_test = (X_test > 0).astype(np.float)

# create, train log regression model and make prediction
log_reg = linear_model.LogisticRegression()
log_reg.fit(X, y)
# predict method return binary labels (0,1) as class for each sample
y_log_label = log_reg.predict(X_test)

# create, train and test linear regression model
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X, y)
y_lin = lin_reg.predict(X_test)

# predict_proba method return a probability of belonging object  to class
y_log_probabilty = log_reg.predict_proba(X_test)[:,1]


# draw source data (points)
plt.figure(1, figsize=(4, 3))
plt.scatter(X.ravel(), y, color='black', zorder=20)

# draw plot for log regression
plt.plot(X_test, y_log_label, color='red', linewidth=3)

# draw plot for linear refression
plt.plot(X_test, y_lin, linewidth=1)

# draw threshold line (all points above this line belong a 1-class,
# all points under line belong 0-class)
plt.axhline(.5, color='green')

# visualize ideal logistic curve (not a model)
plt.plot(X_test, y_log_probabilty, color='blue', linewidth=3)

# define axis and draw plot
plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Label Logistic Regression Model',
            'Linear Regression Model',
            'Probabilty Log Regression'),
           loc="lower right", fontsize='small')
plt.show()


# evaluate model accuracy
lin_score = lin_reg.score(X_test, y_test)
print("Linear regression accuracy: ", lin_score)

log_score = accuracy_score(y_log_label, y_test)
print("Logistic regression accuracy: ", log_score)
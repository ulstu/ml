# -*- coding: utf-8 -*-
import numpy as np


def activation(x):
    return 1 / (1 + np.exp(-x))


def sigma_derivative(x):
    return x * (1 - x)


X = np.array([[0, 0, 1],
              [0.3, 1, 0],
              [1, 0.3, 1],
              [0.6, 0.2, 0],
              [0.6, 0.2, 1]])

y = np.array([[0],
              [1],
              [1],
              [0],
              [0]])

np.random.seed(4)

W_1_2 = 2 * np.random.random((3, 5)) - 1
W_2_3 = 2 * np.random.random((5, 1)) - 1

speed = 1.1

for j in range(100000):
    l1 = X
    l2 = activation(np.dot(l1, W_1_2))
    l3 = activation(np.dot(l2, W_2_3))

    l3_error = y - l3

    if (j % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l3_error)))

    l3_sigma = l3_error * sigma_derivative(l3)

    l2_error = l3_sigma.dot(W_2_3.T)

    l2_sigma = l2_error * sigma_derivative(l2)

    W_2_3 += speed * l2.T.dot(l3_sigma)
    W_1_2 += speed * l1.T.dot(l2_sigma)


X_test = np.array([[0, 0, 0],
                   [0.6, 0.8, 1],
                   [0.6, 0.6, 1],
                   [1, 1, 0],
                   [0.1, 0.1, 0],
                   [0.2, 0.2, 1]])

# target Y_test [0, 1, 1, 1, 0, 0]

l1 = X_test
l2 = activation(np.dot(l1, W_1_2))
l3 = activation(np.dot(l2, W_2_3))
print l3

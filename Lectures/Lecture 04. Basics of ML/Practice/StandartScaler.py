import numpy as np
from sklearn import preprocessing

X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

print("STANDART SCALER")
scaler = preprocessing.StandardScaler().fit(X)
print(scaler)
print("X_scaled mean: ", scaler.mean_)
print("X_scaled std: ", scaler.scale_)

print("X_scaled: ", scaler.transform(X))

# exit()

# The function normalize provides a quick and easy way to perform this operation on
#  a single array-like dataset, either using the l1 or l2 norms:

print("NORMALIZATION")

X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -4.]]
X_normalized = preprocessing.normalize(X, norm='l2', axis=0)
print (X_normalized)


X_normalized = preprocessing.normalize(X, norm='l1', axis=0)
print (X_normalized)

X_normalized = preprocessing.normalize(X, norm='max', axis=0)
print (X_normalized)
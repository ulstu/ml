#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

x,y = make_classification(n_samples=1000, n_features=2,n_redundant=0,n_clusters_per_class=1,n_classes=3)#x是1000行2列;y是1000行1列

x_train=x[:750,:]
x_test=x[750:,:]
y_train=y[:750]
y_test=y[750:]

#plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
#plt.show()

#plt.scatter(x[:, 0], x[:, 1])


# kmeans
plt.figure()


for k in range(3,20,1):
    clf = KMeans(n_clusters=k)
    s = clf.fit(x_train)

    print(s) 
#    print clf.cluster_centers_
#    print clf.labels_
    k_labels=clf.labels_
    print (clf.inertia_)  
    k_inertia=clf.inertia_
    # print clf.predict(x_test)  
    k_pred=clf.predict(x_test)
    plt.plot(k,k_inertia,c='g',marker='x')

plt.show()


for k in range(3,9):
    clf = KMeans(n_clusters=k) 
    s = clf.fit(x_train) 
    numSamples = len(x_train)
    centroids = clf.labels_
    # print centroids,type(centroids)
    print (clf.inertia_) 
    # k_inertia = clf.inertia_
    # k_pred = clf.predict(x_test)
    # plt.plot(k, k_inertia, c='g', marker='x')

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numSamples):
        #markIndex = int(clusterAssment[i, 0])
        plt.plot(x_train[i][0], x_train[i][1], mark[clf.labels_[i]]) #mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    centroids =  clf.cluster_centers_
    for i in range(k):
        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
        #print centroids[i, 0], centroids[i, 1]
    plt.show()

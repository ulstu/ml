# English
# Assignment for practical work 3. Basics of neural networks
## General Assignment

Before performing the practical work, you must download the data set accordingly to your option on the disk
1. Write a program that splits the original sample into a training set, validation set, test set, if such a partition is not provided for by the proposed data set.
2. Scale the features.
3. Using the scikit-learn library (http://scikit-learn.org/stable/), train 2 models of the neural network (Perceptron and MLPClassifier) ​​for the training sample. 
Example MLPClassifier: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
Example and description of Perceptron: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
4. Check the accuracy of the model for the test sample.
5. Conduct experiments and determine the best parameters of the learning rate, the regularization parameter, the optimization function.
The data of the experiments must be presented in the report (graphs, experimental procedure, conclusions).

## Options
Data sets are taken from the UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets.html

The option is determined by the data set, which can be downloaded from the link above:
1. Abalone
2. Adult
3. Artificial Characters
4. ser Knowledge Modeling Data (Students' Knowledge Levels on DC Electrical Machines)
5. EEG Eye State
6. seismic-bumps
7. banknote authentication
8. Weight Lifting Exercises monitored with Inertial Measurement Units
9. REALDISP Activity Recognition Dataset
10. mage Segmentation
11. ISOLET
12. sEMG for Basic Hand movements
13. Letter Recognition
14. Dataset for Sensorless Drive Diagnosis
15. Phishing Websites
16. Multiple Features
17. Diabetic Retinopathy Debrecen Data Set
18. Page Blocks Classification
19. Optical Recognition of Handwritten Digits
20. Pen-Based Recognition of Handwritten Digits
21. Smartphone-Based Recognition of Human Activities and Postural Transitions
22. Indoor User Movement Prediction from RSS data
23. Spambase

With regard to the presentation of experimental results, one possible solution would be:
```
learning_rates = [0.01, 0.04,......]
alphas = [0.01, .......]
optimizers = ['adam', 'lbfgs', ...]
for opt in optimizers:
	for lr in learning_rates:
		for alpha in alphas:
			mlp = MLPClassifier(hidden_layer=[], alhpa = alpha, learning_rate = lr, solver=opt)
			mlp.fit()
			make scoring of model (f1 or r2 or accuracy or ...)
	build 3d plot loss(learning_rates, alphas)
```

# Russian
# Задание на лабораторную работу №3. Основы нейронных сетей
## Общее задание

Перед выполнением лабораторной работы необходимо загрузить набор данных в соответствии с вариантом на диск
1. Написать программу, которая разделяет исходную выборку на обучающую и тестовую (training set, validation set, test set), если такое разделение не предусмотрено предложенным набором данных.
2. Произвести масштабирование признаков (scaling).
3. С использованием библиотеки scikit-learn (http://scikit-learn.org/stable/) обучить 2 модели нейронной сети (Perceptron и MLPClassifier) по обучающей выборке. Перед обучением необходимо осуществить масштабирование признаков. 
Пример MLPClassifier: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
Пример и описание Perceptron: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
4. Проверить точность модели по тестовой выборке.
5. Провести эксперименты и определить наилучшие параметры коэффициента обучения, параметра регуляризации, функции оптимизации.
Данные экспериментов необходимо представить в отчете (графики, ход проведения эксперимента, выводы).

## Варианты
Массивы данных берутся из UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets.html

Вариант определяется набором данных, который можно загрузить по ссылке выше:
1. Abalone
2. Adult
3. Artificial Characters
4. ser Knowledge Modeling Data (Students' Knowledge Levels on DC Electrical Machines)
5. EEG Eye State
6. seismic-bumps
7. banknote authentication
8. Weight Lifting Exercises monitored with Inertial Measurement Units
9. REALDISP Activity Recognition Dataset
10. mage Segmentation
11. ISOLET
12. sEMG for Basic Hand movements
13. Letter Recognition
14. Dataset for Sensorless Drive Diagnosis
15. Phishing Websites
16. Multiple Features
17. Diabetic Retinopathy Debrecen Data Set
18. Page Blocks Classification
19. Optical Recognition of Handwritten Digits
20. Pen-Based Recognition of Handwritten Digits
21. Smartphone-Based Recognition of Human Activities and Postural Transitions
22. Indoor User Movement Prediction from RSS data
23. Spambase











import numpy as np
import math
from sklearn import cross_validation
from sklearn import svm

f = open("../data/winequality-red.csv")
f.readline()  # skip the header

dataset = np.loadtxt(f, delimiter=";")
(N, M) = dataset.shape

# Separate the data from the target attributes
X = dataset[:, 0:(M-1)]
y = dataset[:, (M-1)]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

# Train and test svc all-vs-all(one-vs-one)

clf = svm.SVC(decision_function_shape='ovo')

y_pred = clf.fit(X_train, y_train).predict(X_test)

print("All-vs-All: Number of mislabeled points: %d / %d" % ((y_test != y_pred).sum(), len(X_test)))
print("All-vs-ALL: Root mean squared error: %f" % math.sqrt(((y_test - y_pred)**2).mean()))
print("")


# Train and test svc one-vs-all(one-vs-rest)

clf = svm.SVC(decision_function_shape='ovr')

y_pred = clf.fit(X_train, y_train).predict(X_test)

print("One-vs-All: Number of mislabeled points: %d / %d" % ((y_test != y_pred).sum(), len(X_test)))
print("One-vs-ALL: Root mean squared error: %f" % math.sqrt(((y_test - y_pred)**2).mean()))
print("")

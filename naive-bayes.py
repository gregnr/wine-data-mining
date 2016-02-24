import numpy as np
import math
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


f = open("../data/winequality-red.csv")
f.readline()  # skip the header

dataset = np.loadtxt(f, delimiter=";")
(N, M) = dataset.shape

# Separate the data from the target attributes
X = dataset[:, 0:(M-1)]
y = dataset[:, (M-1)]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

# Train and test Guassian Naive Bayes

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("GaussianNB: Number of mislabeled points: %d / %d" % ((y_test != y_pred).sum(), len(X_test)))
print("GaussianNB: Root mean squared error: %f" % math.sqrt(((y_test - y_pred)**2).mean()))
print("")


# Train and test Multinomial Naive Bayes

mnb = MultinomialNB()

y_pred = mnb.fit(X_train, y_train).predict(X_test)

print("MultinomialNB: Number of mislabeled points: %d / %d" % ((y_test != y_pred).sum(), len(X_test)))
print("MultinomialNB: Root mean squared error: %f" % math.sqrt(((y_test - y_pred)**2).mean()))
print("")


# Train and test Bernoulli Naive Bayes

bnb = BernoulliNB()

y_pred = bnb.fit(X_train, y_train).predict(X_test)

print("BernoulliNB: Number of mislabeled points: %d / %d" % ((y_test != y_pred).sum(), len(X_test)))
print("BernoulliNB: Root mean squared error: %f" % math.sqrt(((y_test - y_pred)**2).mean()))
print("")
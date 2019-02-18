# Logistic regression can also be used to predict the dependent or target variable with multiclass.
from sklearn import datasets
import numpy as np
import pandas as pd
iris = datasets.load_iris()
X = iris.data
y = iris.target
print('Class labels:', np.unique(y))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
# l1 regularization gives better results
lr = LogisticRegression(penalty='l1', C=10, random_state=0)
lr.fit(X_train, y_train)

from sklearn import metrics
# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, lr.predict(X_train)))
print("Train - Confusion matrix :",metrics.confusion_matrix(y_train,lr.predict(X_train)))
print("Train - classification report :", metrics.classification_report(y_train, lr.predict(X_train)))
print("Test - Accuracy :", metrics.accuracy_score(y_test, lr.predict(X_test)))
print("Test - Confusion matrix :",metrics.confusion_matrix(y_test,lr.predict(X_test)))
print("Test - classification report :", metrics.classification_report(y_test, lr.predict(X_test)))

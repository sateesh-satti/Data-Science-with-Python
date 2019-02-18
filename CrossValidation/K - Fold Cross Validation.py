#   K-folds cross-validation splits the training dataset into k-folds without replacement, that
# is, any given data point will only be part of one of the subset, where k-1 folds are used for
# the model training and one fold is used for testing. The procedure is repeated k times so
# that we obtain k models and performance estimates.

# We then calculate the average performance of the models based on the individual
# folds to obtain a performance estimate that is less sensitive to the subpartitioning of the
# training data compared to the holdout or single fold method.

from sklearn.model_selection import cross_val_score,train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import tree
# read the data in
df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Diabetes.csv")
X = df.ix[:,:8].values # independent variables
y = df['class'].values # dependent variables
# Normalize Data
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2017)
# build a decision tree classifier
clf = tree.DecisionTreeClassifier(random_state=2017)

# evaluate the model using 10-fold cross-validation
train_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
test_scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=5)
print("Train Fold AUC Scores: ", train_scores)
print("Train CV AUC Score: ", train_scores.mean())
print("\nTest Fold AUC Scores: ", test_scores)
print("Test CV AUC Score: ", test_scores.mean())


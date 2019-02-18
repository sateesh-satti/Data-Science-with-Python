# An extended cross-validation is the Stratified K-fold cross-validation, where the class
# proportions are preserved in each fold, leading to better bias and variance estimates.

# stratified kfold cross-validation
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import cross_validation
import numpy as np
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

kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
train_scores = []
test_scores = []
for k, (train, test) in enumerate(kfold):
	clf.fit(X_train[train], y_train[train])
	train_score = clf.score(X_train[train], y_train[train])
	train_scores.append(train_score)
	# score for test set
	test_score = clf.score(X_train[test], y_train[test])
	test_scores.append(test_score)

	print('Fold: %s, Class dist.: %s, Train Acc: %.3f, Test Acc: %.3f'
	      % (k + 1 , np.bincount(y_train[train]) , train_score , test_score))

print('\nTrain CV accuracy: %.3f' % (np.mean(train_scores)))
print('Test CV accuracy: %.3f' % (np.mean(test_scores)))

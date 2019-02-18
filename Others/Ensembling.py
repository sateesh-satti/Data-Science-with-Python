# Ensemble methods enable combining multiple model scores into a single score to create
# a robust generalized model.

# At a high level there are two types of ensemble methods.
#   1. Combine multiple models of similar type
#       • Bagging (Bootstrap aggregation)
#       • Boosting

#   2. Combine multiple models of various types
#       • Vote Classification
#       • Blending or Stacking

# Bagging :-

# Bootstrap aggregation (also known as bagging) is a model aggregation technique to reduce model variance.
# The training data is split into multiple samples with replacements called bootstrap samples.
# Bootstrap sample size will be the same as the original sample size, with 3/4th of the original values
# and replacement result in repetition of values.

# Independent models on each of the bootstrap samples are built, and the average of the
# predictions for regression or majority vote for classification is used to create the final model.

# Bagged Decision Trees for Classification
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Diabetes.csv")
X = df.ix[:,:8].values # independent variables
y = df['class'].values # dependent variables
#Normalize
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2017)
kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
num_trees = 100

# Decision Tree with 5 fold cross validation
clf_DT = DecisionTreeClassifier(random_state=2017).fit(X_train,y_train)
results = cross_validation.cross_val_score(clf_DT, X_train,y_train, cv=kfold)
print("Decision Tree (stand alone) - Train : ", results.mean())
print("Decision Tree (stand alone) - Test : ", metrics.accuracy_score(clf_DT.predict(X_test), y_test))

# # Using Bagging Lets build 100 decision tree models and average/majority vote prediction
clf_DT_Bag = BaggingClassifier(base_estimator=clf_DT, n_estimators=num_trees, random_state=2017).fit(X_train,y_train)
results = cross_validation.cross_val_score(clf_DT_Bag, X_train, y_train, cv=kfold)
print("\nDecision Tree (Bagging) - Train : ", results.mean())
print("Decision Tree (Bagging) - Test : ", metrics.accuracy_score(clf_DT_Bag.predict(X_test), y_test))


# Feature Importance : The decision tree model has an attribute to show important features that are based on the
# gini or entropy information gain.

feature_importance = clf_DT.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, df.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# RandomForest :-

# A subset of observations and a subset of variables are randomly picked to build multiple
# independent tree-based models. The trees are more un-correlated as only a subset of
# variables are used during the split of the tree, rather than greedily choosing the best split
# point in the construction of the tree.

from sklearn.ensemble import RandomForestClassifier
num_trees = 100
clf_RF = RandomForestClassifier(n_estimators=num_trees).fit(X_train, y_train)
results = cross_validation.cross_val_score(clf_RF, X_train, y_train, cv=kfold)
print("\nRandom Forest (Bagging) - Train : ", results.mean())
print("Random Forest (Bagging) - Test : ", metrics.accuracy_score(clf_RF.predict(X_test), y_test))

# Extremely Randomized Trees (ExtraTree) :- This algorithm is an effort to introduce more randomness to the bagging process. Tree
# splits are chosen completely at random from the range of values in the sample at each
# split, which allows us to reduce the variance of the model further – however, at the cost of
# a slight increase in bias.

from sklearn.ensemble import ExtraTreesClassifier
num_trees = 100
clf_ET = ExtraTreesClassifier(n_estimators=num_trees).fit(X_train, y_train)
results = cross_validation.cross_val_score(clf_ET, X_train, y_train, cv=kfold)
print("\nExtraTree - Train : ", results.mean())
print("ExtraTree - Test : ", metrics.accuracy_score(clf_ET.predict(X_test), y_test))

# Bagging - Essential Tuning Parameters :
# n_estimators: This is the number of trees, the larger the better. Note that beyond a certain
# point the results will not improve significantly.

# max_features: This is the random subset of features to be used for splitting node,
# the lower the better to reduce variance (but increases bias). Ideally, for a regression
# problem it should be equal to n_features (total number of features) and for classification
# square root of n_features.

# n_ jobs: Number of cores to be used for parallel construction of trees. If set to -1,
# all available cores in the system are used, or you can specify the number.

# Boosting :- The core concept of boosting is that rather
# than an independent individual hypothesis, combining hypotheses in a sequential order
# increases the accuracy.

# Essentially, boosting algorithms convert the weak learners into
# strong learners. Boosting algorithms are well designed to address the bias problems.

# At a high level the AdaBoosting process can be divided into three steps. See Figure 4-7.
# • Assign uniform weights for all data points W0(x) = 1 / N, where N
# is the total number of training data points.
# • At each iteration fit a classifier ym(xn) to the training data and
# update weights to minimize the weighted error function.

# the final model combined model will have a
# minimum error term and maximum learning rate leading to a higher degree of accuracy.

# Bagged Decision Trees for Classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Diabetes.csv")
X = df.ix[:,:8].values # independent variables
y = df['class'].values # dependent variables
#Normalize
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2017)
kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
num_trees = 100

# Decision Tree with 5 fold cross validation
clf_DT = DecisionTreeClassifier(random_state=2017).fit(X_train,y_train)
results = cross_validation.cross_val_score(clf_DT, X_train,y_train, cv=kfold)
print("Decision Tree (stand alone) - Train : ", results.mean())
print("Decision Tree (stand alone) - Test : ", metrics.accuracy_score(clf_DT.predict(X_test), y_test))

# Using Adaptive Boosting of 100 iteration
clf_DT_Boost = AdaBoostClassifier(base_estimator=clf_DT, n_estimators=num_trees, learning_rate=0.1, random_state=2017).fit(X_train,y_train)
results = cross_validation.cross_val_score(clf_DT_Boost, X_train, y_train,cv=kfold)
print("\nDecision Tree (AdaBoosting) - Train : ", results.mean())
print("Decision Tree (AdaBoosting) - Test : ", metrics.accuracy_score(clf_DT_Boost.predict(X_test), y_test))

# GBM - Gradient boosting method

from sklearn.ensemble import GradientBoostingClassifier
# Using Gradient Boosting of 100 iterations
clf_GBT = GradientBoostingClassifier(n_estimators=num_trees, learning_rate=0.1, random_state=2017).fit(X_train, y_train)
results = cross_validation.cross_val_score(clf_GBT, X_train, y_train,cv=kfold)
print("\nGradient Boosting - CV Train : %.2f" % results.mean())
print("Gradient Boosting - Train : %.2f" % metrics.accuracy_score(clf_GBT.predict(X_train), y_train))
print("Gradient Boosting - Test : %.2f" % metrics.accuracy_score(clf_GBT.predict(X_test), y_test))

# GBClassifier for Digits Dataset
from sklearn.ensemble import GradientBoostingClassifier
df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Digit.csv")
X = df.ix[:,1:17].values
y = df['lettr'].values

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2017)
kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
num_trees = 10
clf_GBT = GradientBoostingClassifier(n_estimators=num_trees, learning_rate=0.1, max_depth=3, random_state=2017).fit(X_train, y_train)
results = cross_validation.cross_val_score(clf_GBT, X_train, y_train,cv=kfold)
print("\nGradient Boosting - Train : ", metrics.accuracy_score(clf_GBT.predict(X_train), y_train))
print("Gradient Boosting - Test : ", metrics.accuracy_score(clf_GBT.predict(X_test), y_test))

# Let's predict for the letter 'T' and understand how the prediction
# accuracy changes in each boosting iteration
X_valid= (2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8)

print("Predicted letter: ", clf_GBT.predict(np.array(X_valid).reshape(1,-1)))
# Staged prediction will give the predicted probability for each boosting iteration
stage_preds = list(clf_GBT.staged_predict_proba(np.array(X_valid).reshape(1,-1)))
final_preds = clf_GBT.predict_proba(np.array(X_valid).reshape(1,-1))
# Plot
x = range(1,27)
label = np.unique(df['lettr'])
plt.figure(figsize=(10,3))
plt.subplot(131)
plt.bar(x, stage_preds[0][0], align='center')
plt.xticks(x, label)
plt.xlabel('Label')
plt.ylabel('Prediction Probability')
plt.title('Round One')
plt.autoscale()

plt.subplot(132)
plt.bar(x, stage_preds[5][0],align='center')
plt.xticks(x, label)
plt.xlabel('Label')
plt.ylabel('Prediction Probability')
plt.title('Round Five')
plt.autoscale()

plt.subplot(133)
plt.bar(x, stage_preds[9][0],align='center')
plt.xticks(x, label)
plt.autoscale()

plt.xlabel('Label')
plt.ylabel('Prediction Probability')
plt.title('Round Ten')
plt.tight_layout()
plt.show()

# Gradient boosting corrects the erroneous boosting iteration’s negative impact in subsequent iterations.
# Boosting – Essential Tuning Parameters :-
# Model complexity and over-fitting can be controlled by using correct values for two categories of parameters.
# 1. Tree structure
#    n_estimators: This is the number of weak learners to be built.
#    max_depth: Maximum depth of the individual estimators. The best value depends on the interaction of the input variables.
#    min_samples_leaf: This will be helpful to ensure sufficient number of samples result in leaf.
#    subsample: The fraction of sample to be used for fitting individual models (default=1).
#   Typically .8 (80%) is used to introduce random selection of samples, which, in turn,
#   increases the robustness against over-fitting.
# 2. Regularization parameter
# learning_rate: this controls the magnitude of change in estimators. Lower learning rate is better, which requires
#    higher n_estimators (that is the trade-off).

# Xgboost (eXtreme Gradient Boosting) :-

# Some of the key advantages of the xgboost algorithm are these:
# • It implements parallel processing.
# • It has a built-in standard to handle missing values, which
# means user can specify a particular value different than other
# observations (such as -1 or -999) and pass it as a parameter.
# • It will split the tree up to a maximum depth unlike Gradient
# Boosting where it stops splitting node on encounter of a negative
# loss in the split.

# XGboost has bundle of parameters, and at a high level we can group them into three
# categories. Let’s look at the most important within these categories.
# 1. General Parameters
#   a. nthread – Number of parallel threads; if not given a value
#       all cores will be used.
#   b. Booster – This is the type of model to be run with gbtree
#       (tree-based model) being the default. ‘gblinear’ to be
#       used for linear models
# 2. Boosting Parameters
#   a. eta – This is the learning rate or step size shrinkage
#       to prevent over-fitting; default is 0.3 and it can range
#       between 0 to 1
#   b. max_depth – Maximum depth of tree with default being 6.
#   c. min_child_weight – Minimum sum of weights of all
#       observations required in child. Start with 1/square root of
#       event rate
#   d. colsample_bytree – Fraction of columns to be randomly
#       sampled for each tree with default value of 1.
#   e. Subsample –Fraction of observations to be randomly
#       sampled for each tree with default of value of 1. Lowering
#       this value makes algorithm conservative to avoid overfitting.
#   f. lambda - L2 regularization term on weights with default
#       value of 1.
#   g. alpha - L1 regularization term on weight.
# 3. Task Parameters
#   a. objective – This defines the loss function to be
#       minimized with default value ‘reg:linear’. For binary
#       classification it should be ‘binary:logistic’ and for
#       multiclass ‘multi:softprob’ to get the probability value
#       and ‘multi:softmax’ to get predicted class. For multiclass
#       num_class (number of unique classes) to be specified.
#   b. eval_metric – Metric to be use for validating model
#       performance.

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# read the data in
df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Diabetes.csv")
# Let's use some weak features as predictors
predictors = ['age','serum_insulin']
target = 'class'
# Most common preprocessing step include label encoding and missing value treatment
from sklearn import preprocessing
for f in df.columns:
	if df[f].dtype=='object':
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(df[f].values))
		df[f] = lbl.transform(list(df[f].values))
df.fillna((-999), inplace=True) # missing value treatment
# Let's use some week features to build the tree
X = df[['age','serum_insulin']] # independent variables
y = df['class'].values # dependent variables
#Normalize
X = StandardScaler().fit_transform(X)
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2017)
num_rounds = 100
clf_XGB = XGBClassifier(n_estimators = num_rounds,objective= 'binary:logistic',seed=2017)
# use early_stopping_rounds to stop the cv when there is no score imporovement
clf_XGB.fit(X_train,y_train, early_stopping_rounds=20, eval_set=[(X_test,y_test)], verbose=False)
results = cross_validation.cross_val_score(clf_XGB, X_train,y_train, cv=kfold)
print("\nxgBoost - CV Train : %.2f" % results.mean())
print("xgBoost - Train : %.2f" % metrics.accuracy_score(clf_XGB.predict(X_train), y_train))
print("xgBoost - Test : %.2f" % metrics.accuracy_score(clf_XGB.predict(X_test), y_test))


# DMatrix the internal data structure of xgboost for input data. It is good practice to convert a large
# dataset to DMatrix object to save preprocessing time.

xgtrain = xgb.DMatrix(X_train, label=y_train, missing=-999)
xgtest = xgb.DMatrix(X_test, label=y_test, missing=-999)
# set xgboost params
param = {'max_depth': 3, # the maximum depth of each tree
		'objective': 'binary:logistic'}
clf_xgb_cv = xgb.cv(param, xgtrain, num_rounds,stratified=True,nfold=5,early_stopping_rounds=20,seed=2017)
print ("Optimal number of trees/estimators is %i" % clf_xgb_cv.shape[0])
watchlist = [(xgtest,'test'), (xgtrain,'train')]
clf_xgb = xgb.train(param, xgtrain,clf_xgb_cv.shape[0], watchlist)
# predict function will produce the probability
# so we'll use 0.5 cutoff to convert probability to class label
y_train_pred = (clf_xgb.predict(xgtrain, ntree_limit=clf_xgb.best_iteration) > 0.5).astype(int)
y_test_pred = (clf_xgb.predict(xgtest, ntree_limit=clf_xgb.best_iteration) > 0.5).astype(int)
print("XGB - Train : %.2f" % metrics.accuracy_score(y_train_pred, y_train))
print("XGB - Test : %.2f" % metrics.accuracy_score(y_test_pred, y_test))


# Ensemble Voting

# voting classifier enables us to combine the predictions through majority voting
# from multiple machine learning algorithms of different types, unlike Bagging/Boosting
# where similar types of multiple classifiers are used for majority voting.

import pandas as pd
import numpy as np
# set seed for reproducability
np.random.seed(2017)
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
# currently its available as part of mlxtend and not sklearn
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import train_test_split

df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Diabetes.csv")

X = df.ix[:,:8] # independent variables
y = df['class'] # dependent variables
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2017)

LR = LogisticRegression(random_state=2017)
RF = RandomForestClassifier(n_estimators = 100, random_state=2017)
SVM = SVC(random_state=0, probability=True)
KNC = KNeighborsClassifier()
DTC = DecisionTreeClassifier()
ABC = AdaBoostClassifier(n_estimators = 100)
BC = BaggingClassifier(n_estimators = 100)
GBC = GradientBoostingClassifier(n_estimators = 100)

clfs = []
print('5-fold cross validation:\n')

for clf, label in zip([LR, RF, SVM, KNC, DTC, ABC, BC, GBC],['Logistic Regression',
																'Random Forest',
																'Support Vector Machine',
																'KNeighbors',
																'Decision Tree',
																'Ada Boost',
																'Bagging' ,
																'Gradient Boosting']):
	scores = cross_validation.cross_val_score(clf , X_train , y_train , cv = 5 ,scoring = 'accuracy')
	print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean() , scores.std() , label))
	md = clf.fit(X , y)
	clfs.append(md)
	print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_test) , y_test)))

# Hard Voting and Soft Voting

# Majority voting is also known as hard voting. The argmax of the sum of predicted
# probabilities is known as soft voting
# Parameter ‘weights’ can be used to assign specific
# weights to classifiers. The predicted class probabilities for each classifier are multiplied by
# the classifier weight and averaged. Then the final class label is derived from the highest
# average probability class label.

# Ensemble Voting
clfs = []
print('5-fold cross validation:\n')
ECH = EnsembleVoteClassifier(clfs=[LR, RF, GBC], voting='hard')
ECS = EnsembleVoteClassifier(clfs=[LR, RF, GBC], voting='soft',weights=[1,1,1])

for clf, label in zip([ECH, ECS],['Ensemble Hard Voting','Ensemble Soft Voting']):
	scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5,scoring='accuracy')
	print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(),scores.std(), label))
	md = clf.fit(X, y)
	clfs.append(md)
	print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_test),y_test)))


# Stacking :- In
# stacking initially, you train multiple base models of different types on training/test datasets.
# It is ideal to mix models that work differently (kNN, bagging, boosting, etc.) so that it can
# learn some part of the problem. At level 1, use the predicted values from base models as
# features and train a model that is known as a meta-model, thus combining the learning of
# an individual model will result in improved accuracy. This is a simple level 1 stacking, and
# similarly you can stack multiple levels of different type of models.


# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(2017) # seed to shuffle the train set
df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Diabetes.csv")

X = df.ix[:,:8] # independent variables
y = df['class'] # dependent variables
#Normalize
X = StandardScaler().fit_transform(X)
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=2017)
kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
num_trees = 10
verbose = True # to print the progress

clfs = [KNeighborsClassifier(),
RandomForestClassifier(n_estimators=num_trees, random_state=2017),
GradientBoostingClassifier(n_estimators=num_trees, random_state=2017)]
#Creating train and test sets for blending
dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

print('5-fold cross validation:\n')
for i, clf in enumerate(clfs):
	scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=kfold,	scoring='accuracy')
	print("##### Base Model %0.0f #####" % i)
	print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	clf.fit(X_train, y_train)
	print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_train),	y_train)))
	dataset_blend_train[:,i] = clf.predict_proba(X_train)[:, 1]
	dataset_blend_test[:,i] = clf.predict_proba(X_test)[:, 1]
	print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_test),	y_test)))

print("##### Meta Model #####")
clf = LogisticRegression()
scores = cross_validation.cross_val_score(clf, dataset_blend_train, y_train,cv=kfold, scoring='accuracy')
clf.fit(dataset_blend_train, y_train)
print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_train), y_train)))
print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_test), y_test)))





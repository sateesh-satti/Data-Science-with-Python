# For a given model, you can define a set of parameter values that you would like to try.
# Then using the GridSearchCV function of scikit-learn, models are built for all possible
# combinations of a preset list of values of hyperparameter provided by you, and the best
# combination is chosen based on the cross-validation score.



from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn import metrics
import pandas as pd
seed = 2017
# read the data in
df = pd.read_csv("C:\\Users\\sateesh\\PycharmProjects\\Learning\\MachineLearningWithPython\\4 Machine Learning - Part 2\\Data\\Diabetes.csv")
X = df.ix[:,:8].values # independent variables
y = df['class'].values # dependent variables
#Normalize
X = StandardScaler().fit_transform(X)
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=seed)
kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=seed)
num_trees = 100
clf_rf = RandomForestClassifier(random_state=seed).fit(X_train, y_train)
rf_params = {
'n_estimators': [100, 250, 500, 750, 1000],
'criterion': ['gini', 'entropy'],
'max_features': [None, 'auto', 'sqrt', 'log2'],
'max_depth': [1, 3, 5, 7, 9]
}

# setting verbose = 10 will print the progress for every 10 task completion
grid = GridSearchCV(clf_rf, rf_params, scoring='roc_auc', cv=kfold,verbose=10, n_jobs=-1)
grid.fit(X_train, y_train)
print('Best Parameters: ', grid.best_params_)
results = cross_validation.cross_val_score(grid.best_estimator_, X_train,y_train, cv=kfold)
print("Accuracy - Train CV: ", results.mean())
print("Accuracy - Train : ", metrics.accuracy_score(grid.best_estimator_.predict(X_train), y_train))
print("Accuracy - Test : ", metrics.accuracy_score(grid.best_estimator_.predict(X_test), y_test))


# As the name suggests the RandomSearch algorithm tries random combinations of a range
# of values of given parameters. The numerical parameters can be specified as a range
# (unlike fixed values in GridSearch). You can control the number of iterations of random
# searches that you would like to perform. It is known to find a very good combination in a
# lot less time compared to GridSearch; however you have to carefully choose the range for
# parameters and the number of random search iteration as it can miss the best parameter
# combination with lesser iterations or smaller ranges.

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
# specify parameters and distributions to sample from
param_dist = {'n_estimators':sp_randint(100,1000),
'criterion': ['gini', 'entropy'],
'max_features': [None, 'auto', 'sqrt', 'log2'],
'max_depth': [None, 1, 3, 5, 7, 9]
}
# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist, cv=kfold,
n_iter=n_iter_search, verbose=10, n_jobs=-1, random_state=seed)
random_search.fit(X_train, y_train)
# report(random_search.cv_results_)
print('Best Parameters: ', random_search.best_params_)
results = cross_validation.cross_val_score(random_search.best_estimator_,X_train,y_train, cv=kfold)
print("Accuracy - Train CV: ", results.mean())
print("Accuracy - Train : ", metrics.accuracy_score(random_search.best_estimator_.predict(X_train), y_train))
print("Accuracy - Test : ", metrics.accuracy_score(random_search.best_estimator_.predict(X_test), y_test))

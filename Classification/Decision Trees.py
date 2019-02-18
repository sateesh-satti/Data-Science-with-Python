#   a decision tree is a tree-like structure where
#   internal nodes represent a test on an attribute,
#   each branch represents outcome of a test,
#   and each leaf node represents class label,
#   and the decision is made after computing all attributes.
#   A path from root to leaf represents classification rules. Thus, a decision tree consists of three types of nodes.
#   Root Node, Branch Node, Leaf Node

# Use training data to build a tree generator model, which will determine which
# variable to split at a node and the value of the split. A decision to stop or split again
# assigns leaf nodes to a class. An advantage of a decision tree is that there is no need for
# the exclusive creation of dummy variables.

# The base algorithm is known as a greedy algorithm, in which the
# tree is constructed in a top-down recursive divide-and-conquer
# manner.
# • At start, all the training examples are at the root.
# • Input data is partitioned recursively based on selected attributes.
# • Test attributes at each node are selected on the basis of a heuristic
# or statistical impurity measure like gini, or information gain (entropy).

# Conditions for Stopping Partitioning :-
# • All samples for a given node belong to the same class.
# • There are no remaining attributes for further partitioning – majority voting is employed for classifying the leaf.
# • There are no samples left.

# Default criterion is “gini” as it’s comparatively faster to compute than “entropy”;
# however both measures give almost identical decisions on split.

from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import tree
import sklearn.metrics as metrics
iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
X = iris.data
y = iris.target
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
# split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=0)
clf.fit(X_train, y_train)
# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
print("Train - Confusion matrix :",metrics.confusion_matrix(y_train, clf.predict(X_train)))
print("Train - classification report :", metrics.classification_report(y_train, clf.predict(X_train)))
print("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
print("Test - Confusion matrix :",metrics.confusion_matrix(y_test, clf.predict(X_test)))
print("Test - classification report :", metrics.classification_report(y_test, clf.predict(X_test)))
tree.export_graphviz(clf, out_file='tree.dot')

from sklearn.externals.six import StringIO
import pydot
out_data = StringIO()
tree.export_graphviz(clf, out_file=out_data,
feature_names=iris.feature_names,
class_names=clf.classes_.astype(int).astype(str),
filled=True, rounded=True,
special_characters=True,
node_ids=1,)
graph = pydot.graph_from_dot_data(out_data.getvalue())
graph[0].write_pdf("iris.pdf") # save to pdf


# Key Parameters for Stopping Tree Growth
# One of the key issues with the decision tree is that the tree can grow very large, ending up creating one leaf per observation.
# max_features: maximum features to be considered while deciding each split, default=“None” which means all features will be considered
# min_samples_split: split will not be allowed for nodes that do not meet this number
# min_samples_leaf: leaf node will not be allowed for nodes less than the minimum samples
# max_depth: no further split will be allowed, default=“None”

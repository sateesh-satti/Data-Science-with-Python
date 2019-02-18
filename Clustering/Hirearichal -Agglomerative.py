# Agglomerative clustering is a hierarchical cluster technique that builds nested clusters
# with a bottom-up approach where each data point starts in its own cluster and as we
# move up, the clusters are merged, based on a distance matrix.

# Key Parameters :-
#   n_clusters: number of clusters to find, default is 2.
#   linkage: It has to be one of the following, that is, ward or complete or average,default=ward.
#       The Ward’s method will merge clusters if the in-cluster variance or the sum of square error is a minimum
#       All pairwise distances of both clusters are used in ‘average’ method, and it is less affected by outliers.
#       The ‘complete’ method considers the distance between the farthest elements of two clusters,
#           so it is also known as maximum linkage.

# Heirarchical clusterings results arrangement can be better interpreted with
# dendogram visualization

from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
import pandas as pd
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt
# Agglomerative Cluster
iris = datasets.load_iris()
# Let's convert to dataframe
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['species'])
# let's remove spaces from column name
iris.columns = iris.columns.str.replace(' ','')
X = iris.ix[:,:3] # independent variables
y = iris.species # dependent variable

model = AgglomerativeClustering(n_clusters=3)
model.fit(X)
iris['pred_species'] = model.labels_
print("Accuracy :", metrics.accuracy_score(iris.species, iris.pred_species))
print("Classification report :", metrics.classification_report(iris.species,iris.pred_species))

from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
# generate the linkage matrix
Z = linkage(X, 'ward')
c, coph_dists = cophenet(Z, pdist(X))
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Agglomerative Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
Z,
leaf_rotation=90., # rotates the x axis labels
leaf_font_size=8., # font size for the x axis labels
)
plt.tight_layout()
plt.show()

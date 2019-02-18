# Clustering is an unsupervised learning problem. Key objective is to identify distinct
# groups (called clusters) based on some notion of similarity within a given dataset.

# The most popularly used clustering techniques are k-means (divisive) and hierarchical (agglomerative).

# The key objective of a k-means algorithm is to organize data into clusters such that there
# is high intra-cluster similarity and low inter-cluster similarity.

# An item will only belong to one cluster, not several, that is, it generates a specific number of disjoint,
#  non-hierarchical clusters. K-means uses the strategy of divide and concur, and it is a classic example for
# an expectation maximization (EM) algorithm.

# EM algorithms are made up of two steps:
# the first step is known as expectation(E) and is used to find the expected point associated with a cluster;
# and the second step is known as maximization(M) and is used to improve the estimation of the cluster using knowledge
# from the first step. The two steps are processed repeatedly until convergence is reached.

# K-means is designed for Euclidean distance only

# Limitations of K-means
# • K-means clustering needs the number of clusters to be specified.
# • K-means has problems when clusters are of differing sized,densities, and non-globular shapes.
# • Presence of outlier can skew the results.


from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib import pyplot as plt
iris = datasets.load_iris()
# Let's convert to dataframe
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['species'])
# let's remove spaces from column name
iris.columns = iris.columns.str.replace(' ','')
iris.head()


X = iris.ix[:,:3] # independent variables
y = iris.species # dependent variable
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

model = KMeans(n_clusters=3, random_state=11)
model.fit(X)
print(model.labels_)
iris['pred_species'] = np.choose(model.labels_, [1, 0, 2]).astype(np.int32)
print("Accuracy :", metrics.accuracy_score(iris.species, iris.pred_species))
print("Classification report :", metrics.classification_report(iris.species,iris.pred_species))

# Set the size of the plot
plt.figure(figsize=(10,7))
# Create a colormap
colormap = np.array(['red', 'blue', 'green'])
# Plot Sepal
plt.subplot(2, 2, 1)
plt.scatter(iris['sepallength(cm)'], iris['sepalwidth(cm)'],c=colormap[iris.species.astype(int)], marker='o', s=50)
plt.xlabel('sepallength(cm)')
plt.ylabel('sepalwidth(cm)')
plt.title('Sepal (Actual)')

plt.subplot(2, 2, 2)
plt.scatter(iris['sepallength(cm)'], iris['sepalwidth(cm)'],c=colormap[iris.pred_species.astype(int)], marker='o', s=50)
plt.xlabel('sepallength(cm)')
plt.ylabel('sepalwidth(cm)')
plt.title('Sepal (Predicted)')

plt.subplot(2, 2, 3)
plt.scatter(iris['petallength(cm)'], iris['petalwidth(cm)'],c=colormap[iris.species.astype(int)],marker='o', s=50)
plt.xlabel('petallength(cm)')
plt.ylabel('petalwidth(cm)')
plt.title('Petal (Actual)')

plt.subplot(2, 2, 4)
plt.scatter(iris['petallength(cm)'], iris['petalwidth(cm)'],c=colormap[iris.pred_species.astype(int)],marker='o', s=50)
plt.xlabel('petallength(cm)')
plt.ylabel('petalwidth(cm)')
plt.title('Petal (Predicted)')
plt.tight_layout()
plt.show()


# Finding Value of k ( Num Of Clusters )
# Two methods are commonly used to determine the value of k.
#   • Elbow method
#   • Average silhouette method

# Elbow Method :-
#   Perform k-means clustering on the dataset for a range of value k (for example 1 to 10) and
# calculate the sum of squared error (SSE) or percentage of variance explained for each k.
# Plot a line chart for cluster number vs. SSE and then look for an elbow shape on the line
# graph, which is the ideal number of clusters. With increase in k the SSE tends to decrease
# toward 0. The SSE is zero if k is equal to the total number of data points in the dataset as
# at this stage each data point becomes its own cluster, and no error exists between cluster
# and its center. So the goal with the elbow method is to choose a small value of k that has
# a low SSE, and the elbow usually represents this value. Percentage of variance explained
# tends to increase with increase in k and we’ll pick the point where the elbow shape
# appears.

from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
K = range(1,10)
KM = [KMeans(n_clusters=k).fit(X) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/X.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X)**2)/X.shape[0]
bss = tss-wcss
varExplained = bss/tss*100

kIdx = 10-1
##### plot ###
kIdx = 2

# elbow curve
# Set the size of the plot
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(K, avgWithinSS, 'b*-')
plt.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.subplot(1, 2, 2)
plt.plot(K, varExplained, 'b*-')
plt.plot(K[kIdx], varExplained[kIdx], marker='o', markersize=12,markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
plt.tight_layout()
plt.show()


# Average Silhouette Method
# The silhouette value will range between -1 and 1 and a
# high value indicates that items are well matched within clusters and weakly matched to
# neighboring clusters.

# s(i) = b(i) – a(i) / max {a(i), b(i)}, where a(i) is average dissimilarity
# of ith item with other data points from same cluster, b(i) lowest average
# dissimilarity of i to other cluster to which i is not a member.


from sklearn.metrics import silhouette_score,silhouette_samples
from matplotlib import cm
score = []
for n_clusters in range(2,10):
	kmeans = KMeans(n_clusters=n_clusters)
	kmeans.fit(X)
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_
	score.append(silhouette_score(X, labels, metric='euclidean'))
# Set the size of the plot
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(score)
plt.grid(True)
plt.ylabel("Silouette Score")
plt.xlabel("k")
plt.title("Silouette for K-means")
# Initialize the clusterer with n_clusters value and a random generator
model = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
model.fit_predict(X)
cluster_labels = np.unique(model.labels_)
n_clusters = cluster_labels.shape[0]
# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(X, model.labels_)
plt.subplot(1, 2, 2)
y_lower, y_upper = 0,0
yticks = []
for i, c in enumerate(cluster_labels):
	c_silhouette_vals = silhouette_vals[cluster_labels == c]
	c_silhouette_vals.sort()
	y_upper += len(c_silhouette_vals)
	color = cm.spectral(float(i) / n_clusters)
	plt.barh(range(y_lower, y_upper), c_silhouette_vals, facecolor=color,edgecolor=color, alpha=0.7)
	yticks.append((y_lower + y_upper) / 2)
	y_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.yticks(yticks, cluster_labels+1)
# The vertical line for average silhouette score of all the values
plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.title("Silouette for K-means")
plt.show()

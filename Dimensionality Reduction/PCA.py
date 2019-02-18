# Existence of a large number of features or dimensions makes analysis computationally
# intensive and hard for performing machine learning tasks for pattern identification. PCA
# is the most popular unsupervised linear transformation technique for dimensionality
# reduction. PCA finds the directions of maximum variance in high-dimensional data
# such that most of the information is retained and projects it onto a smaller dimensional
# subspace

# Perform eigen decomposition, that is, compute eigen vectors that
# are the principal component which will give the direction and
# compute eigen values which will give the magnitude.

# Sort the eigen pairs and select eigen vectors with the largest eigen
# values that cumulatively capture information above a certain
# threshold

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
iris = datasets.load_iris()
X = iris.data
# standardize data
X_std = StandardScaler().fit_transform(X)
# create covariance matrix
cov_mat = np.cov(X_std.T)

print('Covariance matrix \n%s' %cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
# sort eigenvalues in decreasing order
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cummulative Variance Explained", cum_var_exp)
plt.figure(figsize=(6, 4))
plt.bar(range(4), var_exp, alpha=0.5, align='center',
label='Individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',
label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# In the above plot we can see that the first three principal components are explaining
# 99% of the variance. Let’s perform PCA using scikit-learn and plot the first three eigen
# vectors.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
# import some data to play with
iris = datasets.load_iris()
Y = iris.target
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y, cmap=plt.
cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()

# Recommender systems are the core information filtering system designed to predict the user preference and help to
# recommend correct items to create a user-specific personalization experience.
# There are two types of recommendation systems:
# 1) content-based filtering and
# 2) collaborative filtering.

# Content-Based Filtering :- This type of system focuses on the similarity attributes of the items to give you recommendations.
# example, so if a user has purchased items of a particular category then other similar items from the same category are
# recommended to the user

# Collaborative Filtering (CF) :- CF focuses on the similarity attribute of the users, that is,
# it finds people with similar tastes based on a similarity measure from the large group of users.
#  There are two types of CF implementation in practice: memory-based and model-based.

# The memory-based recommendation is mainly based on the similarity
# algorithm; the algorithm looks at items liked by similar people to create a ranked list of
# recommendations. You can then sort the ranked list to recommend the top n items to the
# user.

import numpy as np
import pandas as pd
df = pd.read_csv('Data/movie_rating.csv')
n_users = df.userID.unique().shape[0]
n_items = df.itemID.unique().shape[0]
print('\nNumber of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))


# Create user-item similarity matrices
df_matrix = np.zeros((n_users, n_items))
for line in df.itertuples():
	df_matrix[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(df_matrix, metric='cosine')
item_similarity = pairwise_distances(df_matrix.T, metric='cosine')

# Top 3 similar users for user id 7
print("Similar users for user id 7: \n", pd.DataFrame(user_similarity).loc[6,pd.DataFrame(user_similarity).loc[6,:] > 0].sort_values(ascending=False)[0:3])

# Top 3 similar items for item id 6
print("Similar items for item id 6: \n", pd.DataFrame(item_similarity).loc[5,pd.DataFrame(item_similarity).loc[5,:] > 0].sort_values(ascending=False)[0:3])


# Let’s build the user-based predict and item-based prediction for forumla as a
# function. Apply this function to predict ratings and use root mean squared error (RMSE)
# to evaluate the model performance.

# Function for item based rating prediction
def item_based_prediction(rating_matrix, similarity_matrix):
	return rating_matrix.dot(similarity_matrix) / np.array([np.abs(similarity_matrix).sum(axis=1)])

# Function for user based rating prediction
def user_based_prediction(rating_matrix, similarity_matrix):
	mean_user_rating = rating_matrix.mean(axis=1)
	ratings_diff = (rating_matrix - mean_user_rating[:, np.newaxis])
	return mean_user_rating[:, np.newaxis] + similarity_matrix.dot(ratings_diff) / np.array([np.abs(similarity_matrix).sum(axis=1)]).T

item_based_prediction = item_based_prediction(df_matrix, item_similarity)
user_based_prediction = user_based_prediction(df_matrix, user_similarity)

# Calculate the RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, actual):
	prediction = prediction[actual.nonzero()].flatten()
	actual = actual[actual.nonzero()].flatten()
	return sqrt(mean_squared_error(prediction, actual))

print('User-based CF RMSE: ' + str(rmse(user_based_prediction, df_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_based_prediction, df_matrix)))

y_user_based = pd.DataFrame(user_based_prediction)

# Predictions for movies that the user 6 hasn't rated yet
predictions = y_user_based.loc[6,pd.DataFrame(df_matrix).loc[6,:] == 0]
top = predictions.sort_values(ascending=False).head(n=1)
recommendations = pd.DataFrame(data=top)
recommendations.columns = ['Predicted Rating']
print(recommendations)

y_item_based = pd.DataFrame(item_based_prediction)

# Predictions for movies that the user 6 hasn't rated yet
predictions = y_item_based.loc[6,pd.DataFrame(df_matrix).loc[6,:] == 0]
top = predictions.sort_values(ascending=False).head(n=1)
recommendations = pd.DataFrame(data=top)
recommendations.columns = ['Predicted Rating']
print(recommendations)

# Model-based CF is based on matrix factorization (MF) such as Singular Value
# Decomposition (SVD) and Non-nagative matrix factorization (NMF)

# code for recommender system using SVD

# calculate sparsity level
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print('The sparsity level of is ' + str(sparsity*100) + '%')
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# Get SVD components from train matrix. Choose k.
u, s, vt = svds(df_matrix, k = 5)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, df_matrix)))




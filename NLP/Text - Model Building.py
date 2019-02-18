# Text Similarity :- A measure that indicates how similar two objects are is described through a distance
# measure with dimensions represented by features of the objects (here text).

# A smaller distance indicates a high degree of similarity and vice versa.

# For text similarity, it is important to choose the right distance measure to get better results.
# There are various distance measures available and Euclidian metric is the most common,
# which is a straight line distance between two points.

# code for calculating cosine similarity for documents

from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
# Function to create a dictionary with key as file names and values as text for all files in a given folder
def CorpusFromDir(dir_path):
	result = dict(docs = [open(os.path.join(dir_path,f)).read() for f in
	os.listdir(dir_path)],ColNames = map(lambda x: x, os.listdir(dir_path)))
	return result

docs = CorpusFromDir('Data/')

# Initialize
vectorizer = CountVectorizer()
doc_vec = vectorizer.fit_transform(docs.get('docs'))

#create dataFrame
df = pd.DataFrame(doc_vec.toarray().transpose(), index = vectorizer.get_feature_names())

print("Similarity b/w doc 1 & 2: ", cosine_similarity(df['Doc_1.txt'],df['Doc_2.txt']))
print("Similarity b/w doc 1 & 3: ", cosine_similarity(df['Doc_1.txt'],df['Doc_3.txt']))
print("Similarity b/w doc 2 & 3: ", cosine_similarity(df['Doc_2.txt'],df['Doc_3.txt']))


# Text Clustering :-

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

# load data and print topic names
newsgroups_train = fetch_20newsgroups(subset='train')
print(list(newsgroups_train.target_names))

# Sample TOpics to run a clustering alogorithm and examine keywords of each cluster
categories = ['alt.atheism', 'comp.graphics', 'rec.motorcycles']

dataset = fetch_20newsgroups(subset='all', categories=categories,shuffle=True, random_state=2017)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
labels = dataset.target

print("Extracting features from the dataset using a sparse vectorizer")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(dataset.data)
print("n_samples: %d, n_features: %d" % X.shape)


# LSA :- Latent Semantic Analysis
# LSA is a mathematical method that tries to bring out latent relationships within
# a collection of documents. Rather than looking at each document isolated from
# the others, it looks at all the documents as a whole and the terms within them to
# identify relationships.

# “Latent Semantic Analysis (LSA)” and “Latent Semantic Indexing (LSI)” is the same thing,

# code for LSA through SVD

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# Lets reduce the dimensionality to 2000
svd = TruncatedSVD(2000)
lsa = make_pipeline(svd, Normalizer(copy=False))

X = lsa.fit_transform(X)

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

# k-means clustering on SVD dataset

from __future__ import print_function
km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1)

# Scikit learn provides MiniBatchKMeans to run k-means in batch mode suitable for a very large corpus
# km = MiniBatchKMeans(n_clusters=5, init='k-means++', n_init=1, init_size=1000, batch_size=1000)

print("Clustering sparse data with %s" % km)
km.fit(X)

print("Top terms per cluster:")
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(3):
	print("Cluster %d:" % i, end='')
	for ind in order_centroids[i, :10]:
		print(' %s' % terms[ind], end='')
	print()


# Topic Modeling :- Topic modeling algorithms enable you to discover hidden topical patterns or thematic
# structure in a large collection of documents. The most popular topic modeling techniques
# are Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF).

# Latent Dirichlet Allocation (LDA) :-

# LDA’s objective is to maximize separation between means of projected topics and
# minimize variance within each projected topic

# So LDA defines each topic as a bag of words by carrying out three steps described below.
# Step 1: Initialize k clusters and assign each word in the document to one of the k  topics.
# Step 2: Re-assign word to new topic based on a) how is the proportion of words
#    for a document to a topic, and b) how is the proportion of a topic widespread across all documents.
# Step 3: Repeat step 2 until coherent topics result.

from sklearn.decomposition import LatentDirichletAllocation
# continuing with the 20 newsgroup dataset and 3 topics
total_topics = 3
lda = LatentDirichletAllocation(n_topics=total_topics,max_iter=100,learning_method='online',learning_offset=50.,random_state=2017)

lda.fit(X)
feature_names = np.array(vectorizer.get_feature_names())
for topic_idx, topic in enumerate(lda.components_):
	print("Topic #%d:" % topic_idx)
	print(" ".join([feature_names[i] for i in topic.argsort()[:-20 - 1:-1]]))

# Non-negative Matrix Factorization :-
# NMF is a decomposition method for multivariate data, and is given by V = MH, where V
# is the product of matrices W and H. W is a matrix of word rank in the features, and H is
# the coefficient matrix with each row being a feature. The three matrices have no negative
# elements.

from sklearn.decomposition import NMF
nmf = NMF(n_components=total_topics, random_state=2017, alpha=.1, l1_ratio=.5)
nmf.fit(X)
for topic_idx, topic in enumerate(nmf.components_):
	print("Topic #%d:" % topic_idx)
	print(" ".join([feature_names[i] for i in topic.argsort()[:-20 - 1:-1]]))


# Text Classification :-
# categories of topics to retrieve from dataset
categories = ['alt.atheism', 'comp.graphics', 'rec.motorcycles', 'sci.space', 'talk.politics.guns']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=2017, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=2017, remove=('headers', 'footers', 'quotes'))

y_train = newsgroups_train.target
y_test = newsgroups_test.target

vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf = True, max_df=0.5, ngram_range=(1, 2), stop_words='english')
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

print("Train Dataset")
print("%d documents" % len(newsgroups_train.data))
print("%d categories" % len(newsgroups_train.target_names))
print("n_samples: %d, n_features: %d" % X_train.shape)
print("Test Dataset")
print("%d documents" % len(newsgroups_test.data))
print("%d categories" % len(newsgroups_test.target_names))
print("n_samples: %d, n_features: %d" % X_test.shape)

# code text classification using Multinomial naïve Bayes :-

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
clf = MultinomialNB()
clf = clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print('Train accuracy_score: ', metrics.accuracy_score(y_train, y_train_pred))
print('Test accuracy_score: ',metrics.accuracy_score(newsgroups_test.target,y_test_pred))
print("Train Metrics: ", metrics.classification_report(y_train, y_train_pred))
print("Test Metrics: ", metrics.classification_report(newsgroups_test.target,y_test_pred))

# Sentiment Analysis :- The procedure of discovering and classifying opinions expressed in a piece of text (like
# comments/feedback text) is called the sentiment analysis. The intended output of this
# analysis would be to determine whether the writer’s mindset toward a topic, product,
# service etc., is neutral, positive, or negative.

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
data = pd.read_csv('Data/customer_review.csv')

SIA = SentimentIntensityAnalyzer()
data['polarity_score']=data.Review.apply(lambda x:SIA.polarity_scores(x)['compound'])
data['neutral_score']=data.Review.apply(lambda x:SIA.polarity_scores(x)['neu'])
data['negative_score']=data.Review.apply(lambda x:SIA.polarity_scores(x)['neg'])
data['positive_score']=data.Review.apply(lambda x:SIA.polarity_scores(x)['pos'])
data['sentiment']=''
data.loc[data.polarity_score>0,'sentiment']='POSITIVE'
data.loc[data.polarity_score==0,'sentiment']='NEUTRAL'
data.loc[data.polarity_score<0,'sentiment']='NEGATIVE'
data.head()

data.sentiment.value_counts().plot(kind='bar',title="sentiment analysis")
plt.show()



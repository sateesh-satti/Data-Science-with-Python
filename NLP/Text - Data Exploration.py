# TF*IDF is an information retrieval technique that weighs a term’s frequency (TF) and its inverse document frequency (IDF).
# Each word or term has its respective TF and IDF score. The product of the TF and IDF scores of a term is called the TF*IDF
# weight of that term.Put simply, the higher the TF*IDF score (weight), the rarer the term and vice versa.


# Frequency Chart :- This visualization presents a bar chart whose length corresponds to the frequency a
#    particular word occurred

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


words = df.index
freq = df.ix[:,0].sort(ascending=False, inplace=False)
pos = np.arange(len(words))
width=1.0
ax=plt.axes(frameon=True)
ax.set_xticks(pos)
ax.set_xticklabels(words, rotation='vertical', fontsize=9)
ax.set_title('Word Frequency Chart')
ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
plt.bar(pos, freq, width, color='b')
plt.show()

# Word Cloud :- This is a visual representation of text data, which is helpful to get a high-level
# understanding about the important keywords from data in terms of its occurrence.
# ‘WordCloud’ package can be used to generate words whose font size relates to its
# frequency.

from wordcloud import WordCloud
# Read the whole text.
text = open(r'C:\Users\sateesh\PycharmProjects\Learning\MachineLearningWithPython\5 TextMining and recommender systems\Data\text_files\Doc_1.txt').read()
# Generate a word cloud image
wordcloud = WordCloud().generate(text)
# Display the generated image:
# the matplotlib way:

import matplotlib.pyplot as plt
plt.imshow(wordcloud.recolor(random_state=2017))
plt.title('Most Frequent Words')
plt.axis("off")
plt.show()

# Lexical Dispersion Plot :-
# This plot is helpful to determine the location of a word in a sequence of text sentences.
# On the x-axis you’ll have word offset numbers and on the y-axis each row is a
# representation of the entire text and the marker indicates an instance of the word of
# interest.


from nltk import word_tokenize
def dispersion_plot(text, words):
	words_token = word_tokenize(text)
	points = [(x,y) for x in range(len(words_token)) for y in range(len(words)) if words_token[x] == words[y]]
	if points:
		x,y=zip(*points)
	else:
		x=y=()
	plt.plot(x,y,"rx",scalex=.1)
	plt.yticks(range(len(words)),words,color="b")
	plt.ylim(-1,len(words))
	plt.title("Lexical Dispersion Plot")
	plt.xlabel("Word Offset")
	plt.show()

text = 'statistics skills, and programming skills are equally important for \
analytics. statistics skills, and domain knowledge are important for analytics'
dispersion_plot(text, ['statistics', 'skills', 'and', 'important'])

# Co-occurrence Matrix :-
# Calculating the co-occurrence between words in a sequence of text will be helpful
# matrices to explain the relationship between words. A co-occurrence matrix tells us how
# many times every word has co-occurred with the current word.

import statsmodels.api as sm
import scipy.sparse as sp

# default unigram model
count_model = CountVectorizer(ngram_range=(1,1))
docs_unigram = count_model.fit_transform(docs.get('docs'))

# co-occurrence matrix in sparse csr format
docs_unigram_matrix = (docs_unigram.T * docs_unigram)
# fill same word cooccurence to 0
docs_unigram_matrix.setdiag(0)

# co-occurrence matrix in sparse csr format
docs_unigram_matrix = (docs_unigram.T * docs_unigram)
docs_unigram_matrix_diags = sp.diags(1./docs_unigram_matrix.diagonal())
# normalized co-occurence matrix
docs_unigram_matrix_norm = docs_unigram_matrix_diags * docs_unigram_matrix
# Convert to a dataframe
df = pd.DataFrame(docs_unigram_matrix_norm.todense(), index = count_model.
get_feature_names())
df.columns = count_model.get_feature_names()
# Plot
sm.graphics.plot_corr(df, title='Co-occurrence Matrix', xnames=list(df.index))
plt.show()


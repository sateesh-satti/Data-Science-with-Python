# step deals with cleansing the consolidated text to remove noise to ensure efficient
# syntactic, semantic text analysis for deriving meaningful insights from text

# Convert to Lower Case and Tokenize
# Sentence Tokenizing
# Word Tokenizing
# Removing Noise
	#   Numbers:
	#   Punctuation:
	#   Stop words
	#   Whitespace:

# Sentence Tokenizing
import nltk
from nltk.tokenize import sent_tokenize
text='Statistics skills, and programming skills are equally important \
for analytics. Statistics skills, and domain knowledge are important for \
analytics. I like reading books and travelling.'

sent_tokenize_list = sent_tokenize(text)
print(sent_tokenize_list)

import nltk.data
spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
print(spanish_tokenizer.tokenize('Hola. Esta es una frase espanola.'))

import nltk
nltk.download()

# word tokenizing
from nltk.tokenize import word_tokenize
print(word_tokenize(text))
# Another equivalent call method using TreebankWordTokenizer
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize(text))


# Removing Noise
# 	Numbers : Numbers are removed as they may not be relevant and not hold valuable information.
import re
def remove_numbers(text):
	return re.sub(r'\d+', '', text)
text = 'This is a sample English sentence, \n with whitespace and numbers 1234!'
print('Removed numbers: ', remove_numbers(text))

# Punctuation: It is to be removed for better identifying each word and remove punctuation characters from the dataset.

import string
# Function to remove punctuations
def remove_punctuations(text):
	words = nltk.word_tokenize(text)
	punt_removed = [w for w in words if w.lower() not in string.punctuation]
	return " ".join(punt_removed)
print(remove_punctuations('This is a sample English sentence, with punctuations!'))

# Stop words: Words like “the,” “and,” “or” are uninformative and add unneeded noise
# to the analysis. For this reason they are removed

from nltk.corpus import stopwords
# Function to remove stop words
def remove_stopwords(text, lang='english'):
	words = nltk.word_tokenize(text)
	lang_stopwords = stopwords.words(lang)
	stopwords_removed = [w for w in words if w.lower() not in lang_stopwords]
	return " ".join(stopwords_removed)
print(remove_stopwords('This is a sample English sentence'))

# Whitespace: Often in text analytics, an extra whitespace (space, tab, Carriage Return,Line Feed) becomes identified as a word.

# Function to remove whitespace
def remove_whitespace(text):
	return " ".join(text.split())
text = 'This is a sample English sentence, \n with whitespace and numbers 1234!'
print('Removed whitespace: ', remove_whitespace(text))

# Part of Speech (PoS) Tagging :

# PoS tagging is the process of assigning language-specific parts of speech such as nouns,
# verbs, adjectives, and adverbs, etc., for each word in the given text.

# NLTK supports multiple PoS tagging models, and the default tagger is maxent_
# treebank_pos_tagger, which uses Penn (Pennsylvania University) Tree bank corpus.

# The same has 36 possible parts of speech tags, a sentence (S) is represented by the parser as a
# tree having three children: a noun phrase (NP), a verbal phrase (VP), and the full stop (.).
# The root of the tree will be S.

# PoS Tagger Short Description : -
# maxent_treebank_pos_tagger - It’s based on Maximum Entropy (ME) classification
#                              principles trained on Wall Street Journal subset of the
#                               Penn Tree bank corpus.
# BrillTagger - Brill’s transformational rule-based tagger.
# CRFTagger  - Conditional Random Fields.
# HiddenMarkovModelTagger  - Hidden Markov Models (HMMs) largely used to assign
#                               the correct label sequence to sequential data or assess
#                               the probability of a given label and data sequence.
# HunposTagge  - A module for interfacing with the HunPos open source POS-tagger.
# PerceptronTagger  - Based on averaged perceptron technique proposed by Matthew Honnibal.
# SennaTagger  - Semantic/syntactic Extraction using a Neural Network Architecture.
# SequentialBackoffTagger - Classes for tagging sentences sequentially, left to right.
# StanfordPOSTagger  - Researched and developed at Stanford University.
# TnT -  Implementation of ‘TnT - A Statistical Part of Speech Tagger’ by Thorsten Brants.


# PoS, the sentence, and visualize sentence tree
from nltk import chunk
tagged_sent = nltk.pos_tag(nltk.word_tokenize('This is a sample English sentence'))
print(tagged_sent)
tree = chunk.ne_chunk(tagged_sent)
tree.draw() # this will draw the sentence tree


# PoS, the sentence, and visualize sentence tree

# To use PerceptronTagger
from nltk.tag.perceptron import PerceptronTagger
PT = PerceptronTagger()
print(PT.tag('This is a sample English sentence'.split()))

# To get help about tags
nltk.help.upenn_tagset('NNP')

# Stemming :- It is the process of transforming to the root word, that is, it uses an algorithm that removes
# common word endings from English words, such as “ly,” “es,” “ed,” and “s.” For example,
# assuming for an analysis you may want to consider “carefully,” “cared,” “cares,” “caringly”
# as “care” instead of separate words.

# Widely Used Stemming algorithms are :-

# porterStemmer :- Oldest and widely used but computaionally intensive
# LancasterStemmer :- Fast and aggressive but stemmed words are not inuitive for shorter words
# SnowballStemmer :- Improved version of porters stemming algorithm

from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer
# Function to apply stemming to a list of words
def words_stemmer(words, type="PorterStemmer", lang="english", encoding="utf8"):
	supported_stemmers = ["PorterStemmer","LancasterStemmer","SnowballStemmer"]
	if type is False or type not in supported_stemmers:
		return words
	else:
		stem_words = []
		if type == "PorterStemmer":
			stemmer = PorterStemmer()
			for word in words:
				stem_words.append(stemmer.stem(word).encode(encoding))
		if type == "LancasterStemmer":
			stemmer = LancasterStemmer()
			for word in words:
				stem_words.append(stemmer.stem(word).encode(encoding))
		if type == "SnowballStemmer":
			stemmer = SnowballStemmer(lang)
			for word in words:
				stem_words.append(stemmer.stem(word).encode(encoding))
	return b" ".join(stem_words)
words = 'caring cares cared caringly carefully'
print("Original: ", words)
print("Porter: ", words_stemmer(nltk.word_tokenize(words), "PorterStemmer"))
print("Lancaster: ", words_stemmer(nltk.word_tokenize(words), "LancasterStemmer"))
print("Snowball: ", words_stemmer(nltk.word_tokenize(words), "SnowballStemmer"))

# Lemmatization : It is the process of transforming to the dictionary base form. For this you can use
# WordNet, which is a large lexical dbUtils for English words that are linked together by
# their semantic relationships. It works as a thesaurus, that is, it groups words together
# based on their meanings.

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Function to apply lemmatization to a list of words
def words_lemmatizer(text, encoding="utf8"):
	words = nltk.word_tokenize(text)
	lemma_words = []
	wl = WordNetLemmatizer()
	for word in words:
		pos = find_pos(word)
		lemma_words.append(wl.lemmatize(word, pos).encode(encoding))
	return b" ".join(lemma_words)

# Function to find part of speech tag for a word
def find_pos(word):
# Part of Speech constants
# ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
# You can learn more about these at http://wordnet.princeton.edu/wordnet/man/wndb.5WN.html#sect3
# You can learn more about all the penn tree tags at https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
	pos = nltk.pos_tag(nltk.word_tokenize(word))[0][1]
# Adjective tags - 'JJ', 'JJR', 'JJS'
	if pos.lower()[0] == 'j':
		return 'a'
# Adverb tags - 'RB', 'RBR', 'RBS'
	elif pos.lower()[0] == 'r':
		return 'r'
# Verb tags - 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
	elif pos.lower()[0] == 'v':
		return 'v'
# Noun tags - 'NN', 'NNS', 'NNP', 'NNPS'
	else:
		return 'n'

print("Lemmatized: ", words_lemmatizer(words))


# NLTK English WordNet includes approximately 155,287 words and 117000 synonym
# sets. For a given word, WordNet includes/provides definition, example, synonyms (group
# of nouns, adjectives, verbs that are similar), atonyms (opposite in meaning to another), etc

from nltk.corpus import wordnet
syns = wordnet.synsets("good")
print("Definition: ", syns[0].definition())
print("Example: ", syns[0].examples())
synonyms = []
antonyms = []

# Print synonums and antonyms (having opposite meaning words)
for syn in wordnet.synsets("good"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			antonyms.append(l.antonyms()[0].name())


print("synonyms: \n", set(synonyms))
print("antonyms: \n", set(antonyms))

# N-grams :-
# One of the important concepts in text mining is n-grams, which are fundamentally a set of
# co-occurring or continuous sequence of n items from a given sequence of large text. The
# item here could be words, letters, and syllables. Let’s consider a sample sentence and try
# to extract n-grams for different values of n.

# extracting n-grams from sentence

from nltk.util import ngrams
from collections import Counter
# Function to extract n-grams from text
def get_ngrams(text, n):
	n_grams = ngrams(nltk.word_tokenize(text), n)
	return [ ' '.join(grams) for grams in n_grams]

text = 'This is a sample English sentence'
print("1-gram: ", get_ngrams(text, 1))
print("2-gram: ", get_ngrams(text, 2))
print("3-gram: ", get_ngrams(text, 3))
print("4-gram: ", get_ngrams(text, 4))


# The N-gram technique is relatively simple and simply increasing the value of n will
# give us more contexts. It is widely used in the probabilistic language model of predicting
# the next item in a sequence: for example, search engines use this technique to predict/
# recommend the possibility of next character/words in the sequence to users as they type.

# code for extracting 2-grams from sentence and store it in a dataframe

text = 'Statistics skills, and programming skills are equally important for \
analytics. Statistics skills, and domain knowledge are important for analytics'
# remove punctuations
text = remove_punctuations(text)
# Extracting bigrams
result = get_ngrams(text,2)
# Counting bigrams
result_count = Counter(result)

# Converting to the result to a data frame
import pandas as pd
df = pd.DataFrame.from_dict(result_count, orient='index')
df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index and column name
print(df)

# Bag of Words :-

# Bag of words is the method where you count the occurrence of words in a document without
# giving importance to the grammar and the order of words. This can be achieved by
# creating Term Document Matrix (TDM).

# It is simply a matrix with terms as the rows
# and document names as the columns and a count of the frequency of words as the
# cells of the matrix.

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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

# Change column headers to be file names
df.columns = docs.get('ColNames')
print(df)

# Document Term Matrix (DTM) is the transpose of Term Document Matrix. In DTM
# the rows will be the document names and column headers will be the terms. Both are in the
# matrix format and useful for carrying out analysis; however TDM is commonly used due to
# the fact that the number of terms tends to be way larger than the document count. In this
# case having more rows is better than having a large number of columns.



# TF - IDF :- TF-IDF is a good statistical measure to reflect the
# relevance of the term to the document in a collection of documents or corpus.

# TF :- Term Frequency
# IDF :- Inverse Document Frequency

# Term frequency will tell you how frequently a given term appears
# TF = No. Of times term appears in a document / Total no of terms in the document

# Document frequency will tell you how important a term is
# DF = d(no. of documents containing given term)/D(the size of the collection of documents)

# IDF = log(Total no. of Documents/no. of documents with a given term in it)

# sklearn provides provides a function TfidfVectorizer to calculate TFIDF
# for text, however by default it normalizes the term vector using L2
# normalization and also IDF is smoothed by adding one to the document
# frequency to prevent zero divisions.

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
doc_vec = vectorizer.fit_transform(docs.get('docs'))
#create dataFrame
df = pd.DataFrame(doc_vec.toarray().transpose(), index = vectorizer.get_feature_names())

# Change column headers to be file names
df.columns = docs.get('ColNames')
print(df)

# TF*IDF is an information retrieval technique that weighs a term’s frequency (TF) and
# its inverse document frequency (IDF). Each word or term has its respective TF and IDF score.
# The product of the TF and IDF scores of a term is called the TF*IDF weight of that term.
# Put simply, the higher the TF*IDF score (weight), the rarer the term and vice versa.



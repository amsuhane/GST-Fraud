import numpy as np
import pandas as pd
import re
import pickle
import itertools
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt

"""
with open("../data/corpus_GST_Fraud.txt", "rb") as fp:
	corpus = pickle.load(fp)
with open("../data/corpus_GST_Fraud.txt", "rb") as fp:
	corpus_summary = pickle.load(fp)
with open("../data/corpus_key_words_GST_Fraud.txt", "rb") as fp:
	corpus_key_words = pickle.load(fp)
"""

def clean_corpus(corpus):
	"""Removes stop words, lowers the words and 
	removes stopwords

	Parameters
	----------
	corpus : ~list
		The newspaper articles

	Return
	------
	tweet_words : ~list
		The cleaned corpus
	"""

	stop_words = set(stopwords.words('english'))
	tweets=np.array(corpus)
	tweet_words = []
	for tweet in tweets:
		tweet_words.append([w.lower() for w in word_tokenize(tweet) if (not w in stop_words) and (w.isalpha())])
	return tweet_words


def find_target_words(tweet_words, WORD_LIMIT):
	"""Builds the vocabulary and also 
	finds the target words to use in the
	co-occurence network 

	Parameters
	----------
	tweet_words : ~list
		The cleaned corpus

	Return
	------
	vocab : ~dict
		The vocabulary of the corpus
	re_vocab : ~dict
		The modified vocabulary of the corpus
	target_words : ~numpy.ndarray
		The target words for co-occurence network
	word_cnt : ~dict
		The word count
	"""

	word_cnt = {}
	for words in tweet_words:
	    for word in words:
	        if word not in word_cnt:
	            word_cnt[word] = 1
	        else:
	            word_cnt[word] += 1
	    
	word_cnt_df = pd.DataFrame({'word': [k for k in word_cnt.keys()], 'cnt': [v for v in word_cnt.values()]})
	tmp = word_cnt_df[word_cnt_df['cnt'] > WORD_LIMIT]
	tmp.sort_values(by='cnt', ascending=False).plot(kind='bar', x='word', y='cnt', figsize=(15,7), legend=False)
	#plt.show()
	vocab = {}
	target_words = word_cnt_df[word_cnt_df['cnt'] > WORD_LIMIT]['word'].as_matrix()
	for word in target_words:
	    if word not in vocab:
	        vocab[word] = len(vocab)
	re_vocab = {}
	for word, i in vocab.items():
	    re_vocab[i] = word
	return vocab, re_vocab, target_words, word_cnt


def build_jacard_matrix(tweet_words, vocab, target_words):
	"""Builds the combination and jacard matrix

	Parameters
	----------
	tweet_words : ~list
		The cleaned corpus
	vocab : ~dict
		The vocabulary of the corpus
	target_words : ~numpy.ndarray
		The target words for co-occurence network

	Return
	------
	jacard_matrix : ~numpy.ndarray
		The jaccard matrix
	"""

	tweet_combinations = [list(itertools.combinations(words, 2)) for words in tweet_words]
	combination_matrix = np.zeros((len(vocab), len(vocab)))
	for tweet_comb in tweet_combinations:
	    for comb in tweet_comb:
	        if comb[0] in target_words and comb[1] in target_words:
	            combination_matrix[vocab[comb[0]], vocab[comb[1]]] += 1
	            combination_matrix[vocab[comb[1]], vocab[comb[0]]] += 1
	for i in range(len(vocab)):
	    combination_matrix[i, i] /= 2	        
	jaccard_matrix = 1 - distance.cdist(combination_matrix, combination_matrix, 'jaccard')
	return jaccard_matrix

def graph_network(vocab, jaccard_matrix, re_vocab, word_cnt):
	"""Builds the co-occurence network jacard matrix

	Parameters
	----------
	jacard_matrix : ~numpy.ndarray
		The jaccard matrix
	vocab : ~dict
		The vocabulary of the corpus
	re_vocab : ~dict
		The modified vocabulary of the corpus
	word_cnt : ~dict
		The word count

	"""

	nodes = []
	for i in range(len(vocab)):
	    for j in range(i+1, len(vocab)):
	        jaccard = jaccard_matrix[i, j]
	        if jaccard > 0:
	            nodes.append([re_vocab[i], re_vocab[j], word_cnt[re_vocab[i]], word_cnt[re_vocab[j]], jaccard])
	G = nx.Graph()
	G.nodes(data=True)
	remove_list = ["gst", "tax"] + ["he", "if", "within", "without", "information"]
	for pair in nodes:
	    node_x, node_y, node_x_cnt, node_y_cnt, jaccard = pair[0], pair[1], pair[2], pair[3], pair[4]
	    if (node_x in remove_list) or (node_y in remove_list):
	    	pass
	    else:
		    if not G.has_node(node_x):
		        G.add_node(node_x, count=node_x_cnt)
		    if not G.has_node(node_y):
		        G.add_node(node_y, count=node_y_cnt)
		    if not G.has_edge(node_x, node_y):
		        G.add_edge(node_x, node_y, weight=jaccard)
	plt.figure(figsize=(15,15))
	pos = nx.spring_layout(G, k=0.1)
	node_size = [d['count']*100 for (n,d) in G.nodes(data=True)]
	nx.draw_networkx_nodes(G, pos, node_color='cyan', alpha=1.0, node_size=node_size)
	nx.draw_networkx_labels(G, pos, fontsize=14)
	edge_width = [d['weight']*10 for (u,v,d) in G.edges(data=True)]
	nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='black', width=edge_width)
	plt.axis('off')
	plt.show()


def graph(corpus, WORD_LIMIT):
	tweet_words = clean_corpus(corpus)
	vocab, re_vocab, target_words, word_cnt = find_target_words(tweet_words, WORD_LIMIT)
	jaccard_matrix = build_jacard_matrix(tweet_words, vocab, target_words)
	graph_network(vocab, jaccard_matrix, re_vocab, word_cnt)
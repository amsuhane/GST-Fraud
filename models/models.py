import pickle 
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def build_co_occurence_matrix(corpus):
	all_text = [(' ').join(corpus)]
	cv = CountVectorizer(ngram_range=(1,1), stop_words = 'english') # You can define your own parameters
	X = cv.fit_transform(all_text)
	Xc = (X.T * X) # This is the matrix manipulation step
	Xc.setdiag(0) # We set the diagonals to be zeroes as it's pointless to be 1
	names = cv.get_feature_names() # This are the entity names (i.e. keywords)
	df = pd.DataFrame(data = Xc.toarray(), columns = names, index = names)
	df.to_csv('data/to_gephi_cause.csv', sep = ',')
	print("Built and saved the co-occurence_matrix")

#def build_and_graph_co_occurence_network(corpus):

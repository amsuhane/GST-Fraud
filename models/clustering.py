from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

'''
with open("../data/corpus_GST_Fraud.txt", "rb") as fp:
	corpus = pickle.load(fp)
with open("../data/corpus_GST_Fraud.txt", "rb") as fp:
	corpus_summary = pickle.load(fp)
with open("../data/corpus_key_words_GST_Fraud.txt", "rb") as fp:
	corpus_key_words = pickle.load(fp)
'''

documents = ["../data/corpus_GST_Fraud.txt"] 
  
# raw documents to tf-idf matrix: 
vectorizer = TfidfVectorizer(stop_words='english', 
                             use_idf=True, 
                             smooth_idf=True)
# SVD to reduce dimensionality: 
svd_model = TruncatedSVD(n_components=2,         
                         algorithm='randomized',
                         n_iter=10)
# pipeline of tf-idf + SVD, fit to and applied to documents:
svd_transformer = Pipeline([('tfidf', vectorizer), 
                            ('svd', svd_model)])
svd_matrix = svd_transformer.fit_transform(documents)

print(svd_matrix)

# svd_matrix can later be used to compare documents, compare words, or compare queries with documents
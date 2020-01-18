from data_scraping.extract_newspaper import *
from models.models import *
from models.graph_co_occ_network import *
import pickle

def extract_data(online=False):
	#newspaper_urls = scrape_newspaper_links("GST Fraud", online)
	newspaper_urls = scrape_newspaper_links("Practical Challenges to GST", online)
	corpus, corpus_summary, corpus_key_words = scrape_newspaper_articles(newspaper_urls, "Practical Challenges to GST", online)
	return corpus, corpus_summary, corpus_key_words


if __name__=="__main__":
	corpus_to_use = "21 Reasons" #"Normal"
	if(corpus_to_use=="Normal"):
		corpus, corpus_summary, corpus_key_words = extract_data(False)
		#build_co_occurence_matrix(corpus)
		#graph([(' ').join(i) for i in corpus_key_words], 8)
		#graph(corpus, 50)
	elif(corpus_to_use=="21 Reasons"):
		corpus = []
		with open('data/21_reasons.txt', 'rb') as fp:
			corpus = fp.readlines()
		graph([i.decode('ascii')[:-1] for i in corpus], 2)
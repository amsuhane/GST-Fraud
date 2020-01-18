from googlesearch import search 
import newspaper
from newspaper import Article
import pickle

def search_query_google(query):
	"""Searches google for the provided query and 
	returns the url list obtained

	Parameters
	----------
	query : ~string
		The string used for google search
	
	Returns
	-------
	my_results_list : ~list
		List of scraped urls
	"""
	my_results_list = []
	for i in search(query,        # The query you want to run
	                tld = 'com',  # The top level domain
	                lang = 'en',  # The language
	                num = 100,     # Number of results per page
	                start = 0,    # First result to retrieve
	                stop = None,  # Last result to retrieve
	                pause = 2.0,  # Lapse between HTTP requests
	               ):
	    my_results_list.append(i)
	return my_results_list

def extract_articles(newspaper_urls):
	"""	Extracts the articles from the provided
	newspaper urls and returns the article object

	Parameters
	----------
	newspaper_urls : ~list
		List of scraped urls
	
	Returns
	-------
	articles : ~newspaper.Article
		List of scraped articles
	"""

	articles = []
	for url in newspaper_urls:
		try:
			article = Article(url)
			article.download()
			article.parse()
			article.nlp()
			articles.append(article)
		except:
			print(url)
	return articles


def scrape_newspaper_links(query, run_online=False):
	"""	Helper function to run the google scraper

	Parameters
	----------
	query : ~string
		The string used for google search
	run_online : ~bool
		Whether to run in online mode or load 
		the saved data from /data in offline mode

	Returns
	-------
	newspaper_urls_ : ~list
		List of scraped urls
	"""

	if run_online:
		newspaper_urls = search_query_google(query)
		with open('data/newspaper_urls_'+('_').join(query.split(' '))+'.txt', 'wb') as fp:
			pickle.dump(newspaper_urls, fp)
	else: 
		with open('data/newspaper_urls_'+('_').join(query.split(' '))+'.txt', 'rb') as fp:
			newspaper_urls = pickle.load(fp)	
	print("Found " + str(len(newspaper_urls)) + " newspapers links for the query: " + str(query))
	return newspaper_urls


def scrape_newspaper_articles(newspaper_urls, query, run_online=False):
	"""	Helper function to run the article scraper

	Parameters
	----------
	newspaper_urls : ~list
		List of scraped urls	
	query : ~string
		The string used for google search
	run_online : ~bool
		Whether to run in online mode or load 
		the saved data from /data in offline mode

	Returns
	-------
	corpus : ~list
		list of article texts
	corpus_summary : ~list
		list of article summaries
	corpus_key_words : ~list
		list of article key_words	
	"""

	if run_online:
		newspapers = extract_articles(newspaper_urls)
		corpus = [i.text for i in newspapers]
		corpus_summary = [i.summary for i in newspapers]
		corpus_key_words = [i.keywords for i in newspapers]
		with open("data/corpus_"+('_').join(query.split(' '))+".txt", "wb") as fp:
			pickle.dump(corpus, fp)
		with open("data/corpus_summary_"+('_').join(query.split(' '))+".txt", "wb") as fp:
			pickle.dump(corpus_summary, fp)
		with open("data/corpus_key_words_"+('_').join(query.split(' '))+".txt", "wb") as fp:
			pickle.dump(corpus_key_words, fp)
	else:
		with open("data/corpus_"+('_').join(query.split(' '))+".txt", "rb") as fp:
			corpus = pickle.load(fp)
		with open("data/corpus_summary_"+('_').join(query.split(' '))+".txt", "rb") as fp:
			corpus_summary = pickle.load(fp)
		with open("data/corpus_key_words_"+('_').join(query.split(' '))+".txt", "rb") as fp:
			corpus_key_words = pickle.load(fp)
	print("Scraped " + str(len(corpus)) + " newspapers links for the query: " + str(query))
	return corpus, corpus_summary, corpus_key_words

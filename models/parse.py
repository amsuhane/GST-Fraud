import pandas as pd
from sacremoses import MosesDetokenizer
import stanfordnlp
#stanfordnlp.download('en') 
import numpy as np
import pickle

with open("../data/corpus_GST_Fraud.txt", "rb") as fp:
  corpus = pickle.load(fp)
with open("../data/corpus_GST_Fraud.txt", "rb") as fp:
  corpus_summary = pickle.load(fp)
with open("../data/corpus_key_words_GST_Fraud.txt", "rb") as fp:
  corpus_key_words = pickle.load(fp)

def parse_corpus(corpus, run_online=False):
    if(!run_online):
        nlp = stanfordnlp.Pipeline()
        parsed_corpus = []
        for article in corpus[:5]:
            doc = nlp(article)
            parsed_corpus.append(doc)
        parse_extracted = []
        for i in 
        parse_extracted = str(parsed_corpus[0].sentences[0].print_tokens())
    print("Parsed "+str(len(parsed_corpus))+" articles using StanfordCoreNLP")

#extract parts of speech
def extract_pos(doc):
    parsed_text = {'word':[], 'pos':[], 'exp':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
            if wrd.pos in pos_dict.keys():
                pos_exp = pos_dict[wrd.pos]
            else:
                pos_exp = 'NA'
            parsed_text['word'].append(wrd.text)
            parsed_text['pos'].append(wrd.pos)
            parsed_text['exp'].append(pos_exp)
    #return a dataframe of pos and text
    return pd.DataFrame(parsed_text)

import pandas as pd

#extract lemma
def extract_lemma(doc):
    parsed_text = {'word':[], 'lemma':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
            #extract text and lemma
            parsed_text['word'].append(wrd.text)
            parsed_text['lemma'].append(wrd.lemma)
    #return a dataframe
    return pd.DataFrame(parsed_text)

#call the function on doc
extract_lemma(doc)

parse_corpus(corpus[:2])
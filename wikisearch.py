import numpy as np
import requests
import wikipedia
import gensim
import nltk
from nltk.corpus import stopwords
import nltk.collocations
from nltk import FreqDist, word_tokenize
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import re


S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"


#return all links in the specified page
def get_links(TITLE):
    PARAMS = {
        'action': "query",
        'titles': TITLE,
        'prop': "links",
        'pllimit': "max",
        'format': "json",
    }

    R = S.get(url=URL, params=PARAMS)
    return R.json()

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(page):
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    tokens_raw = nltk.regexp_tokenize(page.content, pattern)
    tokens = [word.lower() for word in tokens_raw]

    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    stopwords_list += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return [lemmatize_stemming(word) for word in tokens if word not in stopwords_list and len(word) > 3]

#get the 50 most common words in the specified page
def get_50_most_common(page):
    words_stopped = preprocess(page)

    freqdist = FreqDist(words_stopped)
    return freqdist.most_common(50)


#get the normalized values for the 50 most common words in  the 
#spcified page
def normalized_top_50(page):
    words_stopped = preprocess(page)

    freqdist = FreqDist(words_stopped)
    top_50 = freqdist.most_common(50)
    
    total = sum(freqdist.values())
    
    normalized = []
    for word in top_50:
        normalized_frequency = word[1] / total
        normalized.append((word[0],"{:.4}".format(normalized_frequency)))
    return normalized
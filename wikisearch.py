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



def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(page):
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    tokens_raw = nltk.regexp_tokenize(page, pattern)
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


#Create Word2Vec model from two documents
def make_model(current,target):
    #combine into list
    page_text = current.content.split(".") + target.content.split(".")

    text = []

     #format for word2vec
    for clue in page_text:
        sentence = clue.translate(str.maketrans('','',string.punctuation)).split(' ')
        new_sent = [word.lower() for word in sentence]   
        text.append(new_sent)
    
    #create model
    model = gensim.models.Word2Vec(text,sg=1)
    model.train(text, total_examples=model.corpus_count, epochs=model.epochs)
    return model


#Check links for similarity to target. If no links fall within similarity threshold,
#then select a random link.
def check_links(model,current,target,visited):
    
    #get links from current
    links = current.links
    success = []
    errors = []
    
    #check links against model for relevence to target subject
    for l in links:
        for word in l.split(' '):
            word = word.lower()
            try:
                for targetword in target.split(' '):
                    dist = model.wv.distance(word,targetword.lower())
                    if dist < 0.02 and l not in visited:
                        success.append((l,dist))
                        break
            except KeyError:
                errors.append(word)
    #if related links found, use most related link, otherwise random
    if len(success) > 0:
        success.sort(key=lambda tup: tup[1])
        return success[0][0]
    else:
        skiplist = ['Wikipedia', 'Category']
        title = links[np.random.randint(0,len(links))]
        while any(sub in title for sub in skiplist):
            title = links[np.random.randint(0,len(links))]
        print(title)
        return title
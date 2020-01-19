import numpy as np
import requests
import wikipedia
import gensim
import nltk
import sys
from nltk.corpus import stopwords
import nltk.collocations
from nltk import FreqDist, word_tokenize
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import re


#lemmatize a word - convert to a standard form.
def lemmatize_stemming(text):
    """Lemmatize a word and convert it to a standard form"""
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

#preprocess a single document or page
def preprocess(page):
    """Preprocess a document or page
    Tokenize, remove all stopwords from the document, and lemmatize each word."""
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    tokens_raw = nltk.regexp_tokenize(page, pattern)
    tokens = [word.lower() for word in tokens_raw]

    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    stopwords_list += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return [lemmatize_stemming(word) for word in tokens if word not in stopwords_list and len(word) > 3]

#get the 50 most common words in the specified page
def get_50_most_common(page):
    """Returns the 50 most common words on the page."""
    words_stopped = preprocess(page)

    freqdist = FreqDist(words_stopped)
    return freqdist.most_common(50)


#get the normalized values for the 50 most common words in  the 
#spcified page
def normalized_top_50(page):
    """Returns the 50 most common words from the page with normalized scores."""
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
    """Accepts 2 documents and creates a Word2Vec model from both for comparison."""
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


#testing different version
def make_model_new(path):
    """Accepts 2 documents and creates a Word2Vec model from both for comparison."""
    #combine into list
    page_text = []
    for page in path:
        page_content = page.content.split(".")
        page_text += page_content

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
    """Uses the input model to check the similarity of the links in the current page
    to the title of the target page."""
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
        return title
    

#main program, runs the wikigolf program.
def run_golf(start,target):
    """Runs the main wikigolf program"""
    #set up initial variables
    target_page = wikipedia.page(target)
    closest = ("none",sys.maxsize)
    path = [start]
    visited = set(start)
    title = start
    exit = False
    #i = 0

    #for i in range(100):
    for i in range(50):
        try:
            #test for target page
            if title.lower() == target.lower():
                exit = True
                break
            #get next page
            current = wikipedia.page(title)
            #create model
            model = make_model(current,target_page)
            #get common words in current page
            top_20 = get_50_most_common(current.content)[:20]

            #test for similarity to target page
            for word,freq in top_20:
                try:
                    for target_word in preprocess(target_page.title):
                        dist = model.wv.distance(word,target_word)
                        if dist < 0.0016 and closest[1] < dist:
                                closest = (title, dist)
                except KeyError:
                    pass
            search_success = True
        except wikipedia.exceptions.DisambiguationError:
            search_success = False
        except wikipedia.exceptions.PageError:
            search_success = False

        #get next link
        title = check_links(model,current,target,visited)
        visited.add(title)
        path.append(title)
        if i % 100 == 0:
            print(".",end="")
        #i += 1
        
    if exit:
        return title, exit, i
    else:
        return closest[0], exit, i

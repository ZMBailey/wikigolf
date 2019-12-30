import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify
from themeter.themeter import Themeter


# with open('spam_model.pkl', 'rb') as f:
#     model = pickle.load(f)
app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a prediction of P(spam)."""
    data = request.json
    theme = Themeter()
    topics,keywords,name = theme.find_topics(data['user_input'])
    return jsonify({'topics':topics,'keywords':keywords,'title':name})
#     prediction = model.predict_proba([data['user_input']])
#     prediction_round = round(prediction[0][1], 2)
#     return jsonify({'probability': prediction_round})


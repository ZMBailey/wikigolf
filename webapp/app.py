import random
import wikigolf.wikisearch as ws
from flask import Flask, request, render_template, jsonify


app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a prediction of P(spam)."""
    data = request.json
    title,found = ws.run_golf(data['start'],data['target'])
    return jsonify({'title':title,'found':found})


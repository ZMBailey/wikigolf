import random
import wikigolf.wikisearch as ws
from flask import Flask, request, render_template, jsonify


app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/find_page', methods=['GET', 'POST'])
def find_page():
    """Return a prediction of P(spam)."""
    data = request.json
    title,found, hops = ws.run_golf(data['start'],data['target'])
    return jsonify({'title':title,'found':found, 'hops':hops})


import random
import wikigolf.wikisearch as ws
import wikipedia
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


@app.route('/get_titles', methods=['GET', 'POST'])
def get_titles():
    """Return the titles of the input pages."""
    data = request.json
    try:
        s = wikipedia.page(data['start'])
        t = wikipedia.page(data['target'])
        return jsonify({'start':s.title, 'target':t.title})
    except:
        return jsonify({'start':'not found', 'target':'not found'})


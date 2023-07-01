import os
import pickle


from flask import Flask, request, render_template, redirect
from flask_sqlalchemy import SQLAlchemy

from models.lstm import get_graph
from models.sentiment import get_sentiment_polarity_graph

app = Flask(__name__)

app.config['SQLALCHEMY_DATABSE_URI'] = os.environ.get('SQLALCHEMY_DATABSE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

stock_names = []
with open('./stock_names.pickle', 'rb') as pkl:
    stock_names = pickle.load(pkl)

@app.route('/')
def index():
    return render_template('index.html', **{
        'stock_names': stock_names
    })

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/license')
def license():
    return render_template('license.html')

@app.route('/search')
def search():
    query = request.args.get('q')
    is_valid_stock = query in stock_names
    if query is None:
        return render_template('search.html', **{
            'stock_names': stock_names
        })
    elif is_valid_stock:
        return redirect(f'/stock/{query}')
    else:
        return render_template('search.html', **{
            'stock_names': stock_names,
            'error': f'{query} is not a valid stock symbol'
        })

@app.route('/stock/<name>')
def stock(name):
    # predictions graphs
    columns = [
        'Open',
        'High',
        'Low',
        'Close',
        'Adj Close',
        'Volume'
    ]

    graphs = {}

    for col in columns:
        scaler = None
        with open(f'models/saved/scaler_{col}.pickle', 'rb') as pkl:
            scaler = pickle.load(pkl)
        graphs[col] = get_graph(name, col, scaler)
    
    sentiment_graph_data = get_sentiment_polarity_graph(name)

    return render_template('stock.html', **{
        'name': name,
        'stock_names': stock_names,
        'graphs_data': graphs,
        'sentiment_graph_data': sentiment_graph_data
    })

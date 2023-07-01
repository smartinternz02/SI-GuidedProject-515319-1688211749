import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from matplotlib.figure import Figure
import numpy as np
import nltk

import base64
from io import BytesIO


api_key = "4f8016edb36a4e7ca300ecc95e5ee10a"
buy_threshold = 0.1
colors = ['blue', 'red', 'green', 'orange']

def get_sentiment_polarity_graph(stock_name):
    url = f"https://newsapi.org/v2/everything?q={stock_name}&language=en&apiKey={api_key}"

    # Make a GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        data = response.json()

        # Get the news articles
        articles = data.get('articles', [])
         # Get the news articles
        articles = data.get('articles', [])

        # Perform sentiment analysis on each article headline
        sentiments = []
        sid = SentimentIntensityAnalyzer()
        for article in articles:
            headline = article.get('title', '')
            scores = sid.polarity_scores(headline)
            sentiment = scores['compound']
            sentiments.append(sentiment)

        # Generate a line plot for the sentiments with a unique color
        fig = Figure()
        ax = fig.subplots()
        ax.plot(sentiments)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_title(f'Sentiment Analysis of Stock {stock_name} News Headlines')
        ax.set_xlabel('Headline')
        ax.set_ylabel('Sentiment Polarity')

        # Add markers to indicate buying decision
        buy_markers = [i for i, sentiment in enumerate(sentiments) if sentiment >= buy_threshold]
        ax.plot(buy_markers, np.array(sentiments)[buy_markers], 'go', markersize=8, label='Buy')

        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        
        return data

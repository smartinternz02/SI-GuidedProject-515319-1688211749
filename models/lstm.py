import numpy as np
import pandas as pd
import base64
from io import BytesIO

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

def get_data(stock_name, column, start_time, end_time):
    df = pdr.get_data_yahoo(
        stock_name,
        start=start_time,
        end=end_time
    )
    data = df.filter([column])
    return data

def preprocess_data(data, scaler):
    scaled_data = scaler.transform(data.values)
    X = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X

def predict(X, model, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def get_20_days_n_preds(stock_name, column, scaler, n):
    end_time = datetime.now()
    start_time = datetime(end_time.year - 1, end_time.month, end_time.day)

    model = keras.models.load_model(f'models/saved/model_{column}.h5')

    raw_data = get_data(stock_name, column, start_time, end_time)

    for i in range(n): # predict the next 10 days
        new_date = raw_data.index[-1].to_pydatetime() + timedelta(days=1)
        X = preprocess_data(raw_data[-61:], scaler)
        pred = predict(X, model, scaler)
        raw_data.at[new_date, column] = pred[0][0]

    return raw_data[-30:]

def get_graph(stock_name, column, scaler, n_days=10):
    stock_data = get_20_days_n_preds(stock_name, column, scaler, n_days)
    
    fig = Figure()
    ax = fig.subplots()
    ax.set_title(f'{column} Price Predictions')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel(f'{column} Price USD ($)', fontsize=18)
    ax.plot(stock_data)
    ax.axvline(x = datetime.today(), color='r')
    ax.legend([f'{column} Price'], loc='lower left')

    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return data
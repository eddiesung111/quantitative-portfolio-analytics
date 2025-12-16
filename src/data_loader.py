import yfinance as yf
import pandas as pd

def get_stock_data(tickers, start, end):
    # download stock data from Yahoo Finance
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start=start, end=end)['Close']
    return data

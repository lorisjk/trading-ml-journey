import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def main(): 
    tickers = ["NVDA", "KO"]
    data = get_data(tickers)
    correlation_matrix1, correlation_2 = analyze(data, tickers)
    add_rsi(data, window=14)
    add_MACD(data, fast=12, slow=26, signal=9)
    add_target(data)
    add_lags(data, columns=["RSI", "MACD_histogram", "Log_Return"], lags=[1])
    train_data, test_data = split_data(data, train_size=0.8)

    X_train, y_train = get_features_and_target(train_data, feature_columns=["Vol", "RSI", "MACD_histogram", "Log_Return", "RSI_lag1", "MACD_histogram_lag1", "Log_Return_lag1"], target_column=["Target"])
    X_test, y_test = get_features_and_target(test_data, feature_columns=["Vol", "RSI", "MACD_histogram", "Log_Return", "RSI_lag1", "MACD_histogram_lag1", "Log_Return_lag1"], target_column=["Target"])


def process_ticker(ticker, period="2y", window=20):
    df = yf.download(ticker, period=period)
    df_close = pd.DataFrame(df["Close", ticker])
    df_close["Log_Return"] = np.log(df_close["Close", ticker] / df_close["Close", ticker].shift())
    df_close["Vol"] = df_close["Log_Return"].rolling(window=window).std() * np.sqrt(252)
    return df_close

def get_data(tickers):
    data = {}
    for ticker in tickers:
        data[ticker] = process_ticker(ticker)
    return data


def analyze(data, tickers):
    # Schritt 1: alle Vol-Spalten nebeneinander packen, Spaltennamen = Ticker
    combined1 = pd.concat({ticker: df["Vol"] for ticker, df in data.items()}, axis=1)

    # Schritt 2: Zeilen mit NaN raus
    combined_clean1 = combined1.dropna()

    # Schritt 3: Korrelation berechnen
    correlation_matrix1 = combined_clean1.corr()
    print(correlation_matrix1)


     # Schritt 1: alle Log_Return-Spalten nebeneinander packen, Spaltennamen = Ticker
    combined2 = pd.concat({ticker: df["Log_Return"] for ticker, df in data.items()}, axis=1)

    # Schritt 2: Zeilen mit NaN raus
    combined_clean2 = combined2.dropna()

    # Schritt 3: Korrelation log_return rolling(60) berechnen
    correlation_2 = pd.DataFrame(combined_clean2[tickers[0]].rolling(60).corr(combined_clean2[tickers[1]]))
    correlation_matrix2 = combined_clean2.corr()
    print(correlation_matrix2)

    return correlation_matrix1, correlation_2

def add_rsi(data, window=14):
    for ticker, df in data.items(): 
        delta = df["Close", ticker].diff()
        gain = delta.where(delta>0,0)
        loss = -delta.where(delta<0,0)
        rs = gain.rolling(window=window).mean() / loss.rolling(window=window).mean()
        df["RSI"] = 100 - (100 / (1 + rs))
        print(f"Overbought level for {ticker} as percentage of total days: {(df['RSI'] > 70).mean() * 100:.1f}%")
        print(f"Oversold level for {ticker} as percentage of total days: {(df['RSI'] < 30).mean() * 100:.1f}%")

    return data

def add_MACD(data, fast=12, slow=26, signal=9):
    for ticker, df in data.items(): 
        df["EMA_fast"]=df["Close", ticker].ewm(span=fast, adjust=False).mean()
        df["EMA_slow"]=df["Close", ticker].ewm(span=slow, adjust=False).mean()
        df["MACD_line"]=df["EMA_fast"]-df["EMA_slow"]
        df["Signal_line"]=df["MACD_line"].ewm(span=signal, adjust=False).mean()
        df["MACD_histogram"]=df["MACD_line"]-df["Signal_line"]

    return data

def add_target(data): 
    for ticker, df in data.items(): 
        next_close = df["Close", ticker].shift(-1)
        target = (next_close > df["Close", ticker]).where(next_close.notna())
        df["Target"] = target.astype("Int64")
      

    return data

def split_data(data, train_size=0.8):
    train_data = {}
    test_data = {}
    for ticker, df in data.items():
        split_index = int(len(df) * train_size)
        train_data[ticker] = df.iloc[:split_index]
        test_data[ticker] = df.iloc[split_index:]
    return train_data, test_data

def add_lags(data, columns=["RSI", "MACD_histogram", "Log_Return"], lags=[1]): 
    for ticker, df in data.items(): 
        for column in columns: 
            for lag in lags: 
                df[f"{column}_lag{lag}"] = df[column].shift(lag)

    return data

def get_features_and_target(data, feature_columns=["Vol", "RSI", "MACD_histogram", "Log_Return", "RSI_lag1", "MACD_histogram_lag1", "Log_Return_lag1"], target_column=["Target"]):
    X = {}
    y = {}
    for ticker, df in data.items(): 
        all_columns = list(feature_columns) + list(target_column)
        combined = df[all_columns]
        combined_clean = combined.dropna()
        X[ticker] = combined_clean[feature_columns]
        y[ticker] = combined_clean[target_column]
    return X, y

if __name__ == "__main__":
    main()
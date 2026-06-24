import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def main(): 
    tickers = ["NVDA", "KO"]
    data = get_data(tickers)
    plotting(data)
    analyze(data)


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


def analyze(data):
    # Schritt 1: alle Vol-Spalten nebeneinander packen, Spaltennamen = Ticker
    combined = pd.concat({ticker: df["Vol"] for ticker, df in data.items()}, axis=1)

    # Schritt 2: Zeilen mit NaN raus
    combined_clean = combined.dropna()

    # Schritt 3: Korrelation berechnen
    correlation_matrix = combined_clean.corr()
    print(correlation_matrix)

    return correlation_matrix

def plotting(data):
    plt.figure(figsize=(12, 8))
    for ticker, df in data.items():
        plt.plot(df["Vol"], label=f"{ticker} Vol")
    plt.xlabel("Time")
    plt.ylabel("Vol")
    plt.legend()
    plt.grid(True)
    plt.title("Volatility Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("vol_comparison_24_06.png")
    plt.close()

if __name__=="__main__":
    main()





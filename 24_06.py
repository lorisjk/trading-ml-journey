import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def main(): 
    tickers = ["NVDA", "KO"]
    data = get_data(tickers)
    plotting_vol(data)
    correlation_matrix1, correlation_2 = analyze(data)
    plotting_correlation(correlation_2)
    add_rsi(data, window=14)
    plotting_rsi(data)



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
    correlation_2 = pd.DataFrame(combined_clean2["NVDA"].rolling(60).corr(combined_clean2["KO"]))
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
        df.dropna(inplace=True) 
        print(f"Overbought level for {ticker}: {(df['RSI'] > 70).mean() * 100:.1f}%")
        print(f"Oversold level for {ticker}: {(df['RSI'] < 30).mean() * 100:.1f}%")

    return data



def plotting_vol(data):
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

def plotting_correlation(correlation_2): 
    plt.figure(figsize=(12,8))
    plt.plot(correlation_2, label="60-Day Rolling Correlation")
    plt.xlabel("Time")
    plt.ylabel("60-Day Rolling Correlation for log returns")
    plt.axhline(0, color="black", linewidth=0.8) 
    plt.title("Correlation between NVDA and KO")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("correlation_comparison_25_06.png")
    plt.close()

def plotting_rsi(data):
    plt.figure(figsize=(12,8))
    for ticker, df in data.items(): 
        plt.plot(df["RSI"], label=f"{ticker} RSI")
    plt.xlabel("Time")
    plt.ylabel("RSI")
    plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    plt.title("RSI Comparison")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("rsi_comparison_26_06.png")
    plt.close()

if __name__=="__main__":
    main()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def main():

    data_close=get_data()
    plot_1(data_close)
    plot_2(data_close)

def get_data():
    data=yf.download("META", start="2022-01-01", progress=False)


    #discard unneeded data

    data_close=pd.DataFrame(data["Close", "META"])


    #calculate daily returns
    data_close["Daily_Return"]=data_close["Close", "META"].pct_change()

    #rolling method
    data_close["MA_20"]=data_close["Close", "META"].rolling(window=20).mean()
    data_close["MA_50"]=data_close["Close", "META"].rolling(window=50).mean()

    #calculate historical vol
    data_close["Change"]=np.log(data_close["Close", "META"] / data_close["Close", "META"].shift())

    data_close["Vol"]=data_close["Change"].rolling(window=20).std() * np.sqrt(252)

    return data_close

    #plot data 1
def plot_1(data_close):
    plt.figure(figsize=(12,8))
    plt.plot(data_close["Close", "META"], label="Closing Price META")
    plt.plot(data_close["MA_20"], label="20 day MA of META")
    plt.plot(data_close["MA_50"], label="50 day MA of META")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig("WEEK_1_1.png")
    plt.close()


def plot_2(data_close):
    plt.figure(figsize=(12,8))
    plt.plot(data_close["Vol"], label="20 day Vol")
    plt.plot(data_close["Daily_Return"], label="Daily return")
    plt.xlabel("Time")
    plt.ylabel("Vol")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig("WEEK_1_2.png")
    plt.close()

main()
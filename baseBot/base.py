import pandas as pd
import ccxt
import numpy as np
from datetime import datetime
from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

binance = ccxt.binance()
symbol = 'ETH/USDT'
timeframe = '1h'

SCALERLOCATION = "../scalers/%s_scaler.pickle"%symbol.replace("/","")

def tsToDatetime(input):
    return datetime.fromtimestamp(input/1000)

def getdata(exchange, symbol, timeframe):

    print("\n" + exchange.name + ' ' + symbol + ' ' + timeframe + ' chart:')

    # get a list of ohlcv candles
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    ohlcv = pd.DataFrame(ohlcv,columns=["Timestamp","Open","High","Low","Close","Volume"])
    ohlcv['Timestamp'] = ohlcv['Timestamp'].apply(tsToDatetime)
    ohlcv = ohlcv.set_index("Timestamp")
    return ohlcv

def preprocess(df):
    df["pct_change"] = df["Close"].pct_change()
    df["chg_direction"] = np.sign(df["pct_change"])
    # SMAS
    df["SMA5"] = df["Close"].rolling(window=5).mean()
    df["SMA15"] = df["Close"].rolling(window=15).mean()
    df["SMA30"] = df["Close"].rolling(window=30).mean()
    df["SMA100"] = df["Close"].rolling(window=100).mean()
    # ta stuff
    df = df.fillna(df.median())
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df = df.fillna(df.median())
    return df

def scale(df):
    my_file = Path(SCALERLOCATION)
    colnames = df.columns
    if my_file.is_file():
        scaler = joblib.load(my_file)
        print("loading existing scaler ,",my_file)
        df = scaler.transform(df)
    else:
        print("creatign new scaler ",my_file)
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
        joblib.dump(scaler, my_file)
    df = pd.DataFrame(df,columns=colnames)
    return df

def descale(df):
    my_file = Path(SCALERLOCATION)
    colnames = df.columns
    if my_file.is_file():
        scaler = joblib.load(my_file)
        print("loading existing scaler ,",my_file)
        df = scaler.inverse_transform(df)
    else:
        raise Exception("no scaler existing... please scale first")
    df = pd.DataFrame(df,columns=colnames)
    return df

def trendindicator(df):
    minima = []
    maxima = []

    batch = int(len(df) / 5)
    for i in range(int(len(df)/batch)):
        start = i * batch
        end = (i+1) * batch
        block = list(df["Close"][start:end])
        smallest = min(block)
        smallestpos = block.index(smallest)
        biggest = max(block)
        biggestpos = block.index(biggest)
        minima.append(smallestpos+(i*batch))
        maxima.append(biggestpos+(i*batch))
    
    minfirst = min(minima) < min(maxima)
    crntsig = 0
    if minfirst:
        # starting with downtrend, reaching the first minima
        crntsig = -1
    else:
        # starting with uptrend, reaching first maximum
        crntsig = 1
    
    signals = []
    for i in range(len(df)):
        signals.append(crntsig)
        if i in minima:
            crntsig = 1
        elif i in maxima:
            crntsig = -1
    df["signal"] = signals
    return df

def traintestcontrolsplit(df):
    colnames = df.columns
    Y = df["signal"] # .shift(-1)
    Y = Y.fillna(Y.median())
    df = df.drop("signal",axis=1)
    x, x_control, y, y_control = train_test_split(df, Y, test_size=0.13, shuffle=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.13, shuffle=True)
    return x_train, x_test, x_control, y_train, y_test, y_control

if __name__ == '__main__':
    df = getdata(binance, symbol, timeframe)
    df = preprocess(df)
    df = scale(df)
    df = trendindicator(df)
    print(df.tail())
    x_train, x_test, x_control, y_train, y_test, y_control = traintestcontrolsplit(df)
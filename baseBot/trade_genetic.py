from base import getdata,preprocess,binance,symbol,timeframe,scale,trendindicator
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def oneEpisode(row,multiplicator):
    res = row * multiplicator
    res = np.sum(res)
    return res

def geneticEnvironment(df,unscaleddf):
    bestval = -999999
    bestsetting = []
    for episode in tqdm(range(100)):
        multiplicator = np.random.rand(1,df.shape[1])
        # maximum should be df.shape[1]
        # minimum should be 0
        # middle is df.shape[1]/2
        quarter = int(df.shape[1]/4)
        buythreshold = .75
        sellthreshold = .25
        # theory:
        # if res higher buythreshold buy, if res lower sellthreshold sell
        money = 20000
        nrstocks = 0
        commission = .000125

        for i in range(len(df)):
            res = oneEpisode(df.iloc[i].values,multiplicator)
            crntPrice = unscaleddf.iloc[i]["Close"]
            if res > buythreshold:
                # buy
                howmany = int(money/crntPrice)
                cost = howmany * crntPrice * (1+commission)
                if cost > money:
                    howmany -= 1
                    cost = howmany * crntPrice * (1+commission)
                money -= cost
                nrstocks += howmany

            elif res < sellthreshold:
                # sell
                cost = nrstocks * crntPrice * (1-commission)
                money += cost
                nrstocks = 0
        # print("final value: ",money)
        win = 20000 - money
        if win > bestval:
            bestval = win
            bestsetting = multiplicator
    print("best value: ",bestval)

if __name__ == '__main__':

    df = getdata(binance, symbol, timeframe)
    idx = df.index
    crntPrice = df["Close"][-1]
    df["signal"] = np.sign(df["Close"].pct_change())
    df["signal"] = df["signal"].shift(-1) # bc we want to predict next days value
    df = preprocess(df)
    # manual train test split
    train, test = train_test_split(df, test_size=0.1)
    # print(train.shape,test.shape)
    # scale
    scaler = MinMaxScaler()
    colnames = train.columns
    df = pd.DataFrame(scaler.fit_transform(train),columns=colnames)
    df = df.drop(["Open","High","Low","Close"],axis=1)
    # try genetic engineering
    print(df["signal"].value_counts())
    geneticEnvironment(df,train)


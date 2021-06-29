from base import getdata,preprocess,binance,symbol,timeframe,scale,trendindicator
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def oneEpisode(row,multiplicator):
    res = row * multiplicator 
    res = np.average(res)
    return res

def geneticEnvironment(df,unscaleddf):
    bestval = -999999
    bestsetting = []
    for episode in tqdm(range(100)):
        multiplicator = np.random.rand(1,df.shape[1])
        # print(multiplicator)
        # print("sum multiplicator", np.sum(multiplicator))
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

        # start from different points to validate non-randomeness. 
        # TODO: randomize

        startpoints = [0,quarter*2, quarter*3]
        results = []
        for startpos,start in enumerate(startpoints):
            money = 20000
            nrstocks = 0
            for i in range(len(df)):
                res = oneEpisode(df.iloc[i].values,multiplicator)
                # todo: track min and max of allrowvalues, and take medium for decision 
                crntPrice = unscaleddf.iloc[i]["Close"]
                if res > buythreshold and nrstocks == 0:
                    # buy
                    howmany = int(money/crntPrice)
                    cost = howmany * crntPrice * (1+commission)
                    if cost > money:
                        howmany -= 1
                        cost = howmany * crntPrice * (1+commission)
                    money -= cost
                    nrstocks += howmany

                elif res < sellthreshold and nrstocks > 0:
                    # sell
                    cost = nrstocks * crntPrice * (1-commission)
                    money += cost
                    nrstocks = 0
            # print("final value: ",money)
            win = 20000 - money
            # adapt win to startpoint, e.g. when starting from 0 no multiplier, half *2, 3/4 times 4
            winadapted = win * (startpos+1)
            results.append(winadapted)
        # final calculation for epoch
        print(results,win)
        if np.mean(results) > bestval:
            bestval = np.mean(results)
            bestsetting = [episode,bestval,multiplicator]
            
    print("best value: ",bestval, " episode: ",bestsetting[0])

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


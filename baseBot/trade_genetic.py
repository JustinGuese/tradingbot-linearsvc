from base import getdata,preprocess,binance,symbol,timeframe,scale,trendindicator
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random


def oneEpisode(row,multiplicator):
    res = row * multiplicator 
    res = np.sum(res)
    return res

def geneticEnvironment(df,unscaleddf,NOREPETITIONS=10):
    bestval = -999999
    bestsetting = []
    grand_multiplicator = np.random.rand(1,df.shape[1])[0]
    # grand multiplicator needs to be a multithreading list and the following for loop needs to run in processes
    for k in tqdm(range(NOREPETITIONS)):
        for episode in (range(len(df.columns))):
            # every episode try to change the multiplicator a little bit and see if it changes
            multiplicator = grand_multiplicator.copy()
            mutantvalue =  random.uniform(-2, 2)
            multiplicator[episode] *= mutantvalue

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
            restracker = []
            for startpos,start in enumerate(startpoints):
                money = 20000
                nrstocks = 0
                notrades = 0
                for i in range(start,len(df)):
                    res = oneEpisode(df.iloc[i].values,multiplicator)
                    restracker.append(res)
                    buythreshold = np.percentile(restracker,75)
                    sellthreshold = np.percentile(restracker,25)
                    # print("current res values: mean %.2f, 75pct %.2f, 25pct %.2f"%(np.median(restracker),buythreshold,sellthreshold))
                    crntPrice = unscaleddf.iloc[i]["Close"]
                    if res > buythreshold and nrstocks == 0:
                        # buy
                        howmany = int(money/crntPrice)
                        if howmany >= 0:
                            cost = howmany * crntPrice * (1+commission)
                            if cost > money:
                                howmany -= 1
                                cost = howmany * crntPrice * (1+commission)
                            money -= cost
                            nrstocks += howmany
                            notrades += 1
                        else:
                            print("game lost. no money, only minus! ",money)

                    elif res < sellthreshold and nrstocks > 0:
                        # sell
                        cost = nrstocks * crntPrice * (1-commission)
                        money += cost
                        nrstocks = 0
                        notrades += 1
                #  print("final value: ",money, "no trades: ",notrades)
                win = money - 20000
                
                # justhold = howmany for start price, then how much it is worth in the end
                justhold = int(20000/unscaleddf.iloc[start]["Close"] ) * unscaleddf.iloc[-1]["Close"]
                # print("win vs justhold: ",win,justhold)
                base_win = win - justhold # win compared to just holding
                # adapt win to startpoint, e.g. when starting from 0 no multiplier, half *2, 3/4 times 4
                base_winadapted = base_win * (startpos+1)
                results.append(base_winadapted)
                totalwin = base_winadapted + 20000
            # final calculation for epoch
            
            if np.mean(results) > bestval:
                print("new mutation at position %d! improvement %.2f$.  previous win: %.2f, new win: %.2f"%(episode,np.mean(results)-bestval,bestval,np.mean(results)))
                bestval = np.mean(results)
                bestsetting = [episode,bestval,base_win,totalwin,buythreshold,sellthreshold,multiplicator]
                # if it is better, replace the grand multiplicators value with the current one
                grand_multiplicator[episode] *= mutantvalue
            
    print("Results: Best earnings of %.2f$ in %d days. Episode: %d. Buythreshold: %.2f, Sellthreshold: %.2f"%(bestval,len(df)/24,bestsetting[0],bestsetting[4],bestsetting[5]))

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


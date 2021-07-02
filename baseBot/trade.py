import joblib
from base import getdata,preprocess,binance,symbol,timeframe,scale,trendindicator
from train import trainOnAll,LOOKBACK
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

def loadModel():
    model = joblib.load("./models/linsvc.pickle")
    return model

if __name__ == '__main__':

    df = getdata(binance, symbol, timeframe)
    idx = df.index
    crntPrice = df["Close"][-1]
    df = preprocess(df)
    df = scale(df)
    # df = trendindicator(df) # bc signal is target var
    
    # 
    clf = loadModel()
    pred = clf.predict(df)
    
    prediction = np.median(pred[-LOOKBACK:])
    
    # cnrtPrice = df["Close"].tail(1).values[0]
    
    crntframe = pd.read_csv("./logs/linearsvc.csv",parse_dates=['timestamp'])
    crntframe = crntframe.sort_values(by="timestamp",ascending=False)
    # crntframe = crntframe.set_index("timestamp")
    money = crntframe["money"][0]
    crntstocks = crntframe["nrstocks"][0]
    
    # do trade logic
    commission = .00015
    
    print(money,crntPrice,prediction)
    
    if prediction == 1 and crntstocks == 0:
        print("buy! crntPrice: ",crntPrice)
        howmany = int(money / crntPrice)
        cost = howmany * crntPrice * (1+commission)
        if cost > money:
            howmany -= 1
            cost = howmany * crntPrice * (1+commission)
            if cost > money:
                howmany -= 1
                cost = howmany * crntPrice * (1+commission)
        # do the transaction
        moneyprebuy = money
        money -= cost
        crntstocks += howmany
    elif prediction == -1 and crntstocks == -1:
        print("sell!")
        win = crntstocks * crntPrice * (1-commission)
        money += win
        crntstocks = 0
    
    
    
    # save new values
    # timestamp,money,nrstocks,crntprediction, crntPrice
    totalmoney = money + crntstocks * crntPrice
    addition = pd.DataFrame([[idx[-1],prediction,money,crntstocks,crntPrice,totalmoney]],columns=crntframe.columns)
    print(addition)
    crntframe = crntframe.append(addition)
    
    
    
    crntframe.to_csv("./logs/linearsvc.csv",index=0)
    
    
    print(crntframe)
    

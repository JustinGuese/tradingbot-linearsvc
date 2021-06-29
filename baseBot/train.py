from base import *
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

LOOKBACK = 5

def simulateTrading(df,pred,logging = True):
    global LOOKBACK
    money = 20000
    STARTMONEY = money
    nrstocks = 0
    crntTrend = 0
    commission = .0015
    moneyprebuy = 0
    portfolio = []
    tradehist = []
    for i in range(len(df)):
        crntPrice = df.iloc[i]["Close"]
        predblock = np.median(pred[i-LOOKBACK:i])
        if predblock == 1 and crntTrend != 1:
            crntTrend = 1
            if logging: print("buy!!!!!",i,money)
            howmany = int(money / crntPrice)
            cost = howmany * crntPrice * (1+commission)
            if cost > money:
                howmany -= 1
                cost = howmany * crntPrice * (1+commission)
            # do the transaction
            moneyprebuy = money
            money -= cost
            nrstocks += howmany
            
        elif predblock == -1 and crntTrend != -1:
            crntTrend = -1
            # do the transaction
            win = nrstocks * crntPrice * (1-commission)
            earn = (money+win) - moneyprebuy
            tradehist.append(earn)
            if logging: print("you earned $",round(earn))
            money += win
            if logging: print("sellll!!!!",i,money)
            nrstocks = 0
        portfolio.append(money+nrstocks*crntPrice)
    print("final earnings: ",round(portfolio[-1]-STARTMONEY), " in %d days"%(int(len(df)/24)))
    print("# stats")
    print("no trades: ",len(tradehist))
    print("biggest win: ",max(tradehist))
    print("biggest loss: ",min(tradehist))
    print("mean tradehistory: ", np.mean(tradehist))
    print("median tradehistory: ",np.median(tradehist))
    algo1value = abs(max(tradehist) - np.median(tradehist)) + abs(min(tradehist) - np.median(tradehist))
    # i want this to be as low as possible
    print("custom metric, should be as low as possible: ",algo1value)
    
def trainOnAll(df,y):
    clf = LinearSVC()
    clf.fit(df,y)   
    joblib.dump(clf, "../models/linsvc.pickle")

if __name__ == '__main__':

    df = getdata(binance, symbol, timeframe)
    df = preprocess(df)
    df = scale(df)
    df = trendindicator(df)
    x_train, x_test, x_control, y_train, y_test, y_control = traintestcontrolsplit(df)
    
    # 
    clf = LinearSVC()
    clf.fit(x_train,y_train)
    scr = clf.score(x_test,y_test)
    pred = clf.predict(x_test)
    target_names = ['sell','buy']
    print(x_train.tail())
    print(classification_report(y_test, pred, target_names=target_names))
    print(scr)

    # prepare for simulation
    df_nosig = df.drop("signal",axis=1)
    pred = clf.predict(df_nosig)
    dfnormal = descale(df_nosig)
    simulateTrading(dfnormal,pred)
    # save final model
    y = df["signal"]
    trainOnAll(df_nosig,y)
    

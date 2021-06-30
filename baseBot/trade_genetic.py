from base import getdata,preprocess,binance,symbol,timeframe,scale,trendindicator
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random
from multiprocessing import Pool,Value,Array
from datetime import datetime
import csv   


multiplicator,df,unscaleddf = None,None,None

def oneEpisode(row,multiplicator):
    res = row * multiplicator 
    res = np.sum(res)
    return res

def oneRun(episode,noChange = False):
    global multiplicator,df,unscaleddf
    bestval = -99999999999999999
    bestsetting = []
    # every episode try to change the multiplicator a little bit and see if it changes
    if not noChange:
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
        # print("new mutation at position %d! improvement %.2f$.  previous win: %.2f, new win: %.2f"%(episode,np.mean(results)-bestval,bestval,np.mean(results)))
        bestval = np.mean(results)
        bestsetting = [episode,bestval,base_win,totalwin,buythreshold,sellthreshold,multiplicator]
    return bestval,episode,multiplicator[episode]
        
        
        
def geneticEnvironment(NOREPETITIONS=10):
    global multiplicator,df,unscaleddf
    multiplicator = np.random.rand(1,df.shape[1])[0]
    noimprovementcounter = 0
    
    # grand multiplicator needs to be a multithreading list and the following for loop needs to run in processes
    tasks = range(len(df.columns))
    for k in tqdm(range(NOREPETITIONS)):
        # calculate pre mutation result
        baselinebestval,_,_ = oneRun(0,noChange = True)
        
        # create pool
        pool = Pool()
        bestvals,episodes,improvvalue = zip(*pool.map(oneRun,tasks))
        # should return results of every combination, check which achieved a higher value than vanilla
        improvement = bestvals > baselinebestval
        improvementpos = [i for i, x in enumerate(improvement) if x]
        print("we have %d improvements"%len(improvementpos))
        backup_multiplicator = multiplicator.copy()
        for i in improvementpos:
            improvemenent_value = improvvalue[i]
            allele_to_improve = episodes[i]
            multiplicator[allele_to_improve] = improvemenent_value
        hashmult = np.sum(multiplicator)
        # calculate final improvement
        mutationbestval,_,_ = oneRun(0,noChange = True)
        changedollar = mutationbestval - baselinebestval
        if changedollar < 0:
            print("## Generation %d - genetic mutation rejected! Negative win of %.2f$"%(k+1,changedollar))
            noimprovementcounter += 1
            multiplicator = backup_multiplicator
        else:
            print("## Generation %d: Improved setup with %.2f$.   from %.2f$ earnings to %.2f$ earnings. sum of multiplicator: %.2f"%(k+1,changedollar,baselinebestval,mutationbestval,hashmult))
        
        pool.close()
    # save final values
    finalvals = [datetime.now(),mutationbestval]
    finalvals.extend(multiplicator)
    with open(r'../geneResults/results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(finalvals)
        
def main():
    global multiplicator,df,unscaleddf
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
    unscaleddf = train
    # try genetic engineering
    print(df["signal"].value_counts())
    geneticEnvironment()

if __name__ == '__main__':
    main()


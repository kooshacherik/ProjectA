
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


import pandas as pd
import matplotlib.pyplot as plt
import seaborn

import math
from scipy.signal import savgol_filter

import os




#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

#pvalue : MacKinnon`s approximate, asymptotic p-value based on MacKinnon (1994)
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            #if math.isnan(pvalue):
            #    print('its nan')
            #    pvalue = 1.0
            #print(pvalue)
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue

            
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def BestPair(DATA,INSTUMENTS) :

    scores, pvalues, pairs= find_cointegrated_pairs(DATA)
  
    #---------->(P-Values Heat Map)<----------
    #seaborn.heatmap(pvalues, xticklabels=INSTUMENTS, 
    #            yticklabels=INSTUMENTS, cmap='RdYlGn_r' 
    #            , mask = (pvalues >= 0.98)
    #            )
    #plt.show()
    #plt.pause(3)
    #plt.close()        
    
    BestPairPValue = np.max((np.ones(pvalues.shape) - pvalues))
    BestPairPValueIndex = np.where((np.ones(pvalues.shape)) - pvalues == BestPairPValue)
    if len(BestPairPValueIndex) == 0:
        return ['','', -1]
  
    return [INSTUMENTS[BestPairPValueIndex[0][0]],INSTUMENTS[BestPairPValueIndex[1][0]] ,BestPairPValue ]







#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def pythag(pt1, pt2):
    a_sq = (pt2[0] - pt1[0]) ** 2
    b_sq = (pt2[1] - pt1[1]) ** 2
    return math.sqrt(a_sq + b_sq)


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


#---------->(Source Code From a Course )<----------
def local_min_max(pts):
    local_min = []
    local_max = []
    prev_pts = [(0, pts[0]), (1, pts[1])]
    for i in range(1, len(pts) - 1):
        append_to = ''
        #I have added 4 =`s for both inequlities
        if pts[i-1] >= pts[i] <= pts[i+1]:
            append_to = 'min'
        elif pts[i-1] <= pts[i] >= pts[i+1]:
            append_to = 'max'
        if append_to:
            if local_min or local_max:
                prev_distance = pythag(prev_pts[0], prev_pts[1]) * 0.5
                curr_distance = pythag(prev_pts[1], (i, pts[i]))
                if curr_distance >= prev_distance:
                    prev_pts[0] = prev_pts[1]
                    prev_pts[1] = (i, pts[i])
                    if append_to == 'min':
                        local_min.append((i, pts[i]))
                    else:
                        local_max.append((i, pts[i]))
            else:
                prev_pts[0] = prev_pts[1]
                prev_pts[1] = (i, pts[i])
                if append_to == 'min':
                    local_min.append((i, pts[i]))
                else:
                    local_max.append((i, pts[i]))
    return local_min, local_max


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def AllMaxMin(data, mean):
    local_min, local_max = local_min_max(data)
    MaxI=[]
    MinI=[]
    MAX=[]
    MIN=[]
    for l in local_max:

            if l[1] > mean:
                MaxI.append(l[0])
                MAX.append(l)


    for L in local_min:
        if L[1] < mean:
                MinI.append(L[0])
                MIN.append(L)
    
         
    #---------->(PLOT ALL MAX AND MINS OF OUR DATASET)<----------            
    #plt.plot(data)
    #plt.axhline(mean,c='r')
    #plt.plot(MaxI,data[MaxI],'o')
    #plt.plot(MinI,data[MinI],'o')
    #plt.title("Data All Extremums")
    ##plt.plot(MaxI,'o')
    ##plt.plot(MinI,'o')
    
    #plt.show()
    ##plt.pause(1)
    ##plt.close()

    return [MAX,MIN,MaxI,MinI]


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------



def SilScore(Data,kmax = 10):
    sil = []

    for k in range(2, kmax):

        kmeans = KMeans(n_clusters = k).fit(Data)
        labels = kmeans.labels_
        score = silhouette_score(Data, labels, metric = 'euclidean')
        sil.append(score)

    if len(sil)==0:
        return len(Data)
    return sil.index(max(sil))+2


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
def Centroids(whatever,Data,mean):
    if len(whatever) == 0:
        print('No Centroid')
        return []

    #Best K for K-Means
    BestK = SilScore(whatever,len(whatever))

    #K-Means on Maximums And Minimums
    kmeans = KMeans(n_clusters=BestK)
    kmeans.fit(whatever)
    centroids = kmeans.cluster_centers_

    lables = kmeans.labels_

    #---------->(PLOT Centroid for Data)<----------
    #plt.plot(Data) 
    #for w in whatever : 
    #    plt.axhline(w[1])
    #plt.axhline(mean,c='r')
    #plt.show()
    #plt.pause(1)
    #plt.close()

    return centroids





#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def EntrySignal (currPrice, SuppCentroids, RessCentroids,mean ,Range,EligCentroid,CentroidNeighbor):
    distance = 100000
    nearest = 0
    if currPrice > mean :
        for s in SuppCentroids:
            if distance > abs(s[1]- currPrice):
                distance = abs(s[1]- currPrice)
                nearest = s[1]
            if abs(s[1] - mean) <= (1/EligCentroid) *Range:
                continue

            if s[1] <= currPrice+(abs(s[1] - mean)/CentroidNeighbor) and  s[1] >= currPrice - (abs(s[1] - mean)/CentroidNeighbor):

                return ('SellShort',s[1])

    else :
        for s in RessCentroids:
            if distance > abs(s[1]- currPrice):
               distance = abs(s[1]- currPrice)
               nearest = s[1]
            if abs(s[1] -mean) <= (1/EligCentroid)*Range:
                continue

            if s[1] <= currPrice+(abs(s[1] - mean)/CentroidNeighbor) and  s[1] >= currPrice -(abs(s[1] - mean)/CentroidNeighbor):
                return ('BuyLong',s[1])
    return ("NoAction",nearest)

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def BETA(Currency):

    SupDude = Currency.rolling(10).mean()
    #https://tradingstrategyguides.com/beta-trading-strategies-forex/
    return (np.std(SupDude)/(np.std(Currency)))

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def TradeAmount (perTrade , RatioFromTheEquality):
    B = perTrade/(RatioFromTheEquality+1)
    A = RatioFromTheEquality * B
    return [A,B]

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def TotalReturn(All , LorG , FeesAndComisions):
    t = All + LorG - FeesAndComisions
    print('Total Return withou profits: '+str(t))
    return t


def WinLossAmount(WinMoney , LostMoney):
    #PNL ASSYMETRIC
    t = WinMoney - LostMoney
    print('Difference between lose nd win : ')
    if t < 0 :
        print('lost')
        print(t)
    elif t>0:
        print('won')
        print(t)
    else:
        print("both zero or equal")
    return 

def ExtremeWinLoss():
    #CAN HAVE LONG FAT TAILE PNL
    #
    return 

def VarOfReturns():
    #skill or luck
    return
#def EVALUATION():
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def CurrentOrdersInf(DataSet, Orders):
    CurrentPricesOfOrders = []
    CurrentRatioPrices =   []
    for o in Orders:
        pa = o[7]
        pb = o[8]
        CurrentPricesOfOrders.append([DataSet[pa+'.csv'].iloc[-1],DataSet[pb+'.csv'].iloc[-1]])
        #---***---***---***#
        d =DataSet[pa+'.csv']/DataSet[pb+'.csv']
        currPrice = d.iloc[-1]
        CurrentRatioPrices.append(currPrice)
    return (CurrentPricesOfOrders,CurrentRatioPrices)
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def positionCheck (Orders, currentPairPrices,currentRatioPrices,CurrentIndex,PipDictAll,BetaEval,RangeSliceAroundMean):
    w = 0
    L = 0
    wMoney = []
    lMoney = []
    removeOrders = []
    OpenCloseIndexes = []
    l = len(Orders)

    for i in range(l):
        SScurr = Orders[i][0]
        mean   = Orders[i][1]
        SL     = Orders[i][2]
        Range  = Orders[i][3]

        currA  = Orders[i][5]
        currB  = Orders[i][6]

        Pa     = Orders[i][7]
        Pb     = Orders[i][8]

        TradeIndex = Orders[i][9]
        
        PosType = Orders[i][10]

        VolA = Orders[i][11]
        VolB = Orders[i][12]

        BetaA = Orders[i][13]
        BetaB = Orders[i][14]

        RangeA = Orders[i][15]
        RangeB = Orders[i][16]

        EX = Orders[i][17]

        NcurrA = currentPairPrices[i][0]
        NcurrB = currentPairPrices[i][1]

        SSNewCurr = currentRatioPrices[i]


        Direction = SSNewCurr - SScurr
        PipbyRatio = abs(SSNewCurr-SScurr)

        if SSNewCurr <= mean + ((1/RangeSliceAroundMean) * Range) and SSNewCurr >= mean - ((1/RangeSliceAroundMean )* Range):
            w += 1
            wMoney.append( PipbyRatio)

            if PosType:
                [resA , pipA] = PnLCheck(currA,NcurrA,'s')
                [resB , pipB] = PnLCheck(currB,NcurrB,'b')
            else:

                [resA , pipA] = PnLCheck(currA,NcurrA,'b')
                [resB , pipB] = PnLCheck(currB,NcurrB,'s')
            

            PipDictAll[Pa].append([Orders[i][4],resA,pipA,VolA])
            PipDictAll[Pb].append([Orders[i][4],resB,pipB,VolB])   
            
            BetaEval[Orders[i][4]] = [(VolA, resA,(pipA/RangeA)),(VolB, resB,(pipB/RangeB)),PosType]



           
            print('order number '+str(Orders[i][4])+' is closed SUCCESSFULLY')
            OpenCloseIndexes.append([TradeIndex , CurrentIndex,Pa,Pb,mean,Range,SScurr,SSNewCurr,currA,currB,NcurrA,NcurrB,Orders[i][4],VolA,VolB,EX,SL,PosType])

            removeOrders.append(Orders[i][4])

        elif PipbyRatio >= SL : 
            L += 1
         #condition 2vom bayad in beshe ke age az mean dar jahate dorost rad karde bud . na hatman ndazeye stoploss ta

            if PosType  :
                [resA , pipA] = PnLCheck(currA,NcurrA,'s')
                [resB , pipB] = PnLCheck(currB,NcurrB,'b')

                PipDictAll[Pa].append([Orders[i][4],resA,pipA,VolA])
                PipDictAll[Pb].append([Orders[i][4],resB,pipB,VolB])
                
                
                BetaEval[Orders[i][4]] = [(VolA, resA,(pipA/RangeA)),(VolB, resB,(pipB/RangeB)),PosType]

                if Direction < 0 :

                    removeOrders.append(Orders[i][4])
                    print("This order surprisingly end up wining quiet good actually")

                    #Vol LAZEME?
                    OpenCloseIndexes.append([TradeIndex , CurrentIndex,Pa,Pb,mean,Range,SScurr,SSNewCurr,currA,currB,NcurrA,NcurrB,Orders[i][4],VolA,VolB,EX,SL,PosType])
                    L -= 1
                    w += 1
                    wMoney.append( PipbyRatio)
                    continue
            else:

                
                [resA , pipA] = PnLCheck(currA,NcurrA,'b')
                [resB , pipB] = PnLCheck(currB,NcurrB,'s')

                PipDictAll[Pa].append([Orders[i][4],resA,pipA,VolA])
                PipDictAll[Pb].append([Orders[i][4],resB,pipB,VolB])

                BetaEval[Orders[i][4]] = [(VolA, resA,(pipA/RangeA)),(VolB, resB,(pipB/RangeB)),PosType]
                
                if Direction > 0 :
                    
                   
                    removeOrders.append(Orders[i][4])
                    print("This order surprisingly end up wining quiet good actually")
                    OpenCloseIndexes.append([TradeIndex , CurrentIndex,Pa,Pb,mean,Range,SScurr,SSNewCurr,currA,currB,NcurrA,NcurrB,Orders[i][4],VolA,VolB,EX,SL,PosType])
                    L -= 1
                    w += 1
                    wMoney.append( PipbyRatio)
                    continue
           
            lMoney.append(PipbyRatio)
            print('order number '+str(Orders[i][4])+' is closed with LOST')
            OpenCloseIndexes.append([TradeIndex , CurrentIndex,Pa,Pb,mean,Range,SScurr,SSNewCurr,currA,currB,NcurrA,NcurrB,Orders[i][4],VolA,VolB,EX,SL,PosType])
            removeOrders.append(Orders[i][4])

        else :
            continue

    #---------->(Remove Closed Orders)<----------
    for r in removeOrders:
        for O in Orders:
            if O[4]==r:
                Orders.remove(O)
                continue

    return [w,L,wMoney,lMoney,OpenCloseIndexes]



#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def PnLCheck(Curr,NCurr,TradeType):
    Pip = NCurr - Curr
    if TradeType=='s':
        if Pip<0:
            return ['p',Pip]
        else:
            return ['l',Pip]
    else:
        if Pip<0:
            return ['l',Pip]
        else:
            return ['p',Pip]

    
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------      

def RANGE(data):
    max = np.max(data)
    min = np.min(data)
    return max - min

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def WinPercentage(Wins,Loses):
    if Loses == 0:
        print('100% win')
        return -1
    d = (Wins/(Wins+Loses))*100
    print("Win Percentage : "+str(d))
    print('=====================================================================================================================================================================')
    print('=====================================================================================================================================================================')
    return d
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
def PipTable(PipDictAll):
    finalPips ={}
    for pda in PipDictAll:
        print(PipDictAll[pda])
        pnl = 0
        i=0
        for o in PipDictAll[pda]:
            i +=1
            if o[1] == 'p':
                print(pnl)
                print(abs(o[2]))
                print('_______________')
                pnl += abs(o[2])
                print(pnl)
                print ('--------------')
            else:
                print(pnl)
                print(abs(o[2]))
                print('_______________')
                pnl -= abs(o[2])
                print(pnl)
                print ('--------------')
            
        finalPips[pda] = (pnl,i)

    print("{:<10} {:<30} {:<20}".format('Pair' , 'TotalPip', 'NumberOfTrades'))
    for K,V in finalPips.items():
        print("{:<10} {:<30} {:<20}".format(K, V[0],V[1]))
    return finalPips
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
def BetaTable (BetaEval):
    print("{:<10} {:<30} {:<10} {:<30} {:<40}".format('OrderNumber','Vol', 'result','Normaled Pip','TradeType'))
    #print("{:<10} {:<30} {:<10} {:<30} {:<30}{:<10}{:<30}{:<40}".format('OrderNum','Vol A', 'resA','NormalPipA', 'VolB', 'resB','NormalPipB','TradeType'))

    for k in BetaEval:
        Val = BetaEval[k]
        VolA = Val[0][0]
        resA = Val[0][1]
        PipRangeA = Val[0][2]

        VolB = Val[1][0]
        resB = Val[1][1]
        PipRangeB = Val[1][2]
        
        print('=====================================================================================================================================================================')
        print
        print("{:<10} {:<30} {:<10} {:<30}{:<40}".format(k, VolA, resA, PipRangeA, Val[2]))
        print("{:<10} {:<30} {:<10} {:<30}{:<40}".format(k, VolB, resB, PipRangeB, Val[2]))
    print('=====================================================================================================================================================================')
    print('===================== ================================================================================================================================================')


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
def CompareEvals(Instruments , ListOfEvaluations , TopNumber = 5):

    TopN={}
    l = len(ListOfEvaluations)
    
    for ins in Instruments :
        SortList = []
        for E in ListOfEvaluations:
            Number             = E[0]
            (P1,P2,P3,P,P5,P6) = E[1]
            WP                 = E[2]
            BetaEvals          = E[3]
            FinalPips          = E[4]
            SortList.append((Number,(P1,P2,P3,P,P5,P6),FinalPips[ins]))

        
        Sorted = sort(SortList)
        TopN[ins] = Sorted[:6]
    for k in TopN :

        print("INSTRUMENT : "+k)

        print("{:<15} {:<6}{:<5}{:<5}{:<5}{:<5}{:<8} {:<30} {:<30}".format('Machine Number','Window','SL','WMN','SE','SExN','CoolDown','pnl','Number Of Trades'))
        for i in TopN[k]:
            Number = i[0]
            (P1,P2,P3,P4,P5,P6) = i[1]
            (pip , n) = i[2]
            print("{:<15} {:<6}{:<5}{:<5}{:<5}{:<5}{:<8} {:<30} {:<30}".format(Number,P1,P2,P3,P4,P5,P6,pip,n))

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def last(n):
    return n[-1] 
  
def sort(tuples):
    return sorted(tuples, key=last, reverse= True)
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
  

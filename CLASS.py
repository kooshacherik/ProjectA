
import fxcmpy
import datetime as dt
import TradeMachineMethods as TMM
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
import math
import re

class TradeMachine(object):
    
    Columns = ['Name','Date','Time','open','high','low','close','volume']
   
    
    def __init__(self, FolderName,AllFiles,Instruments, CloseColumn, Window, SLCoef, RangeSliceAroundMean, EligibleCentroidCoef, CentroidSliceAroundItself, CoolDown,  BalanceMin):
        self.FolderName = FolderName
        self.Instruments = Instruments
        self.AllFiles = AllFiles
        self.CloseColumn = CloseColumn
        self.Window = Window      
        self.SLCoef = SLCoef
        self.RangeSliceAroundMean = RangeSliceAroundMean
        self.EligibleCentroidCoef = EligibleCentroidCoef
        self.CentroidSliceAroundItself = CentroidSliceAroundItself
        self.PipDictAll = {}
        self.BetaEval = {}
        self.ZZ = 0
        self.Z = {}
        self.ZFname =''
        self.CoolDown= CoolDown
        self.DataPortion = int()
        self.wons=0
        self.losts=0
        self.BalanceMin = BalanceMin
        

        
    
    def ReadFiles(self,Files):
        df={}
        for f in Files:

            df[f]= pd.read_csv(f)
            self.PipDictAll[re.sub(self.FolderName+"\\\|.csv",'',f)] = []

            self.ZZ += 1
            if self.ZZ == 1 :
                self.Z = pd.read_csv(f)
                self.ZFname  = f
                self.DataPortion = len(self.Z)
        return df

    def Trade(self): 
        df = self.ReadFiles(self.AllFiles)
        j = 1
        window = self.Window
        numberOfTrades = 0
        OrdersList=[]
        Balance = 10000
        perTrade= 1

        CoolDownDict = {}
        CoolDown = self.CoolDown

        TotalBreakFlag= False

        

        #---------->(Window Frame PLOT)<----------
        #Z = pd.Series(df[ZFname] .iloc[:,7])
        #plt.ion()
        #figure, ax = plt.subplots(figsize=(10, 8))

        #Background = ax.plot(Z)
        #lof = ax.axvline(0)
        #rof = ax.axvline(100)


        for i in range(980 , self.DataPortion):
            print('======================================================================')
            print('Round Number :'+str(j))
            j += 1 
            print('Going through data number '+str(i-window+1)+' to data number '+str(i))
            
            DataSet={}
            DS={}


            #---------->(Window Frame PLOT(1/3))<----------
            #lof.set_visible(False)
            #rof.set_visible(False)
            for D in df :
                #---------->(Window Frame PLOT(2/3))<----------
                #if D == ZFname:
                #    #DRAWING WINDOW FRAME
                #    lof = ax.axvline(i-window+1,c='r')
                #    rof = ax.axvline(i,c='r')
        
                #    figure.canvas.draw()
                #    figure.canvas.flush_events()
 
        

                d = D.replace(self.FolderName+'\\','')

                DS[d] = pd.Series(df[D].iloc[:,self.CloseColumn])
                DataSet[d] = pd.Series(df[D].iloc[i-window+1:i+1,self.CloseColumn])

                #---------->(Window Check)<----------
                if window > len(DataSet[d]):
                    TotalBreakFlag = True
    
            #---------->(Window Check)<----------    
            if TotalBreakFlag :
                print("Not Enough Data for our window frame of : "+str(window))
                break



            DataSet = pd.DataFrame.from_dict(DataSet)
            DS = pd.DataFrame.from_dict(DS)


            #---------->(Cool Down Process)<---------- 
            CoolDownFinishedKeys=[]
            for key in CoolDownDict : 
                CoolDownDict[key] -= 1
                if CoolDownDict[key] == 0:
                    CoolDownFinishedKeys.append(key)
                    
            for dk in CoolDownFinishedKeys:
                del CoolDownDict[dk]






            #---------->(Current Information For Open Positions)<----------
            (CurrentPricesOfOrders,CurrentRatioPrices) = TMM.CurrentOrdersInf (DataSet , OrdersList)

            #---------->(Position Check)<----------
            NumberOfOpenOrders = len(OrdersList)
            print("Open Positions : "+str(NumberOfOpenOrders))
            if NumberOfOpenOrders != 0:
                w,l,WM,LM,OCI = TMM.positionCheck(OrdersList,CurrentPricesOfOrders,CurrentRatioPrices,i,self.PipDictAll,self.BetaEval,self.RangeSliceAroundMean)
                self.wons += w
                self.losts += l



            ##---------->(Closed Position Trade Charts)<----------
            #    for oci in OCI: 
            #        pa =DS[oci[2]+'.csv']
            #        pb =DS[oci[3]+'.csv']
            #        d = pa/pb
            #        mean = oci[4]
            #        R = oci[5]/self.RangeSliceAroundMean

            #        UpBound = (oci[1]+ window)
            #        LowBound = (oci[0]-window)
            #        HEY = d.iloc [LowBound:UpBound]

                  

                    



            #        print('plots for order number : '+str(oci[12]))
            #        #---------->(Ratio Chart)<----------
                   
            #        plt.figure(1)
            #        plt.plot(HEY)
            #        plt.axhline(mean+R,c='y')
            #        plt.axhline(mean,c='r')
            #        plt.axhline(mean-R,c='y')

            #        plt.axhline(oci[15],c='m')
            #        plt.axhline(oci[15]+(abs(oci[15]-mean)/self.CentroidSliceAroundItself),c='m')
            #        plt.axhline(oci[15]-(abs(oci[15]-mean)/self.CentroidSliceAroundItself),c='m')
                    
            #        plt.plot([oci[0],oci[1]],[oci[6],oci[7]],'y')
            #        plt.plot([oci[0],oci[1]],[HEY[oci[0]],HEY[oci[1]]],'r')
            #        plt.show()
            #        #---------->(Numerator PairA Chart)<----------
            #        pa = pa.iloc [(oci[0]-window) : (oci[1]+ window) ]
            #        plt.figure(2)
            #        plt.plot(pa)
            #        plt.ylabel(oci[2])

            #        plt.plot([oci[0],oci[1]],[oci[8],oci[10]],'k')
            #        plt.plot([oci[0],oci[1]],[pa[oci[0]],pa[oci[1]]],'y')
            #        plt.show()
            #        #---------->(Denominator PairB Chart)<----------
            #        pb = pb.iloc [(oci[0]-window) : (oci[1]+ window) ]
            #        plt.figure(3)
            #        plt.plot(pb)
            #        plt.ylabel(oci[3])
                    
            #        plt.plot([oci[0],oci[1]],[oci[9],oci[11]],'k')
            #        plt.plot([oci[0],oci[1]],[pb[oci[0]],pb[oci[1]]],'y')
            #        plt.show()


            #---------->(Balance Check)<----------
            if Balance < self.BalanceMin:
                print("Not Enough Balance")
                continue    
                                                                                 
            #---------->(Find Best Pairs )<----------                                                    
            #---------->(And Check if there isn`t any )<----------
            [P1 , P2 , PVAL] = TMM.BestPair(DataSet,self.Instruments)
            if PVAL == -1 :
                print('No Good Pairs')
                continue
            
            #---------->(Cool Down Check)<----------
            if P1+P2 in CoolDownDict    :
                print(CoolDownDict[P1+P2])
                print(P1+' and '+P2+ "Still Cooling Down")
                continue


            PairA= DataSet[P1+'.csv']
            PairB= DataSet[P2+'.csv']
            DATA = PairA/PairB

            RangePa = TMM.RANGE(PairA)
            RangePb = TMM.RANGE(PairB)
            Range = TMM.RANGE(DATA)



            DATA = DATA.to_numpy()
            PA = PairA.to_numpy()
            PB = PairB.to_numpy()


            CurrA   =   PairA.iloc[-1]
            CurrB   =   PairB.iloc[-1]
            currPrice = DATA[-1]


            MEAN = DATA.mean()
            MEANA = PairA.mean()
            MEANB = PairB.mean()




            #---------->(Finding All Extremums)<----------
            [local_max , local_min,MaxI,MinI]  = TMM.AllMaxMin(DATA, MEAN)


    
            #---------->(Centroids Of S&R)<----------
            SuppCentroids = TMM.Centroids(local_max,DATA,MEAN)
            RessCentroids = TMM.Centroids(local_min,DATA,MEAN)


            #---------->(PLOT All Centroids And Mean)<----------
            #plt.plot(DATA)

            #for u in SuppCentroids:
            #    plt.axhline(u[1])

            #for u in RessCentroids:
            #    plt.axhline(u[1])

            #plt.axhline(MEAN,c='r')

            #plt.show()

            
            
  
            (SignalMsg, TheExtremum) = TMM.EntrySignal (currPrice , SuppCentroids, RessCentroids, MEAN,Range,self.EligibleCentroidCoef,self.CentroidSliceAroundItself)

            print("Action is : "+SignalMsg)

            if SignalMsg != 'NoAction':
                CoolDownDict[P1+P2] = CoolDown
                print(CoolDownDict)
                print(P1+' and '+P2+' : Cool Down begins')

              
    
            if SignalMsg == 'SellShort':

                BetaA = TMM.BETA(PairA)
                BetaB = TMM.BETA(PairB)

        
                ratio = (BetaA*CurrA)/(BetaB*CurrB)

                #---------->(Amount for each position base on their Beta Coefficient)<----------
                [VolA,VolB] = TMM.TradeAmount(perTrade,ratio)

                Balance -= (perTrade)
                numberOfTrades += 1

                stopLoss =  abs((currPrice - MEAN) * self.SLCoef)
                OrdersList.append([currPrice,MEAN ,stopLoss,Range,numberOfTrades,CurrA,CurrB,P1,P2,i,True,VolA,VolB,BetaA,BetaB,RangePa,RangePb,TheExtremum])

            elif SignalMsg == 'BuyLong':
       
                BetaA = TMM.BETA(PairA)
                BetaB = TMM.BETA(PairB)       
                ratio = (BetaA*CurrA)/(BetaB*CurrB)

                #---------->(Amount for each position base on their Beta Coefficient)<----------
                [VolA,VolB] = TMM.TradeAmount(perTrade,ratio)
        
                Balance -= (perTrade)
                numberOfTrades += 1

                stopLoss = abs((currPrice - MEAN) * self.SLCoef)
                OrdersList.append([currPrice,MEAN ,stopLoss,Range,numberOfTrades,CurrA,CurrB,P1,P2,i,False,VolA,VolB,BetaA,BetaB,RangePa,RangePb,TheExtremum])


        #---------->(Window Frame PLOT(3/3))<----------
        #plt.show()

    def Evaluation(self):
       
        WinP = TMM.WinPercentage(self.wons,self.losts)
        TMM.BetaTable(self.BetaEval)
        print('=====================================================================================================================================================================')
        print('=====================================================================================================================================================================')
        finalPips = TMM.PipTable(self.PipDictAll)
        print('=====================================================================================================================================================================')

        return (( self.Window, self.SLCoef, self.RangeSliceAroundMean, self.EligibleCentroidCoef, self.CentroidSliceAroundItself, self.CoolDown) ,WinP , self. BetaEval , finalPips)

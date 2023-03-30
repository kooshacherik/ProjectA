
This is a summery of my pair trade strategy . [Pair trade](https://en.wikipedia.org/wiki/Pairs_trade) is a [mean reversion](https://en.wikipedia.org/wiki/Mean_reversion_(finance)) strategy where our highest profits are when price is as far as possible from mean , and since we expect price`s inevitable return toward mean, we can make profit if we have bought or sold accordingly . Here is the pipe line :
![PipeLine](https://user-images.githubusercontent.com/76734519/228828233-e3243fd2-377e-46fa-985e-d19752481275.png)




We will investigate among 18 Forex pairs and update to the best possible pair at each time unit .We will pick a window amount slice of each instrument for every time unit and select(if exist)  the most two [cointegrated pairs by Engle-Granger Two-Step Method](https://corporatefinanceinstitute.com/resources/data-science/cointegration/) . Lets call them pair A and pair B .  As they are cointegrated , their ratio which is RatioData = A/B is changing around a constant amount which is its mean . Below heatmap is illustrating cointegration between instruments for a specific time interval with green color and how much each pair is cointegrated with others.
![HeatMap](https://user-images.githubusercontent.com/76734519/228843179-c1a01357-619f-44c0-bb11-21d2985603b2.png)



We will open and close positions by analyzing RatioData. When RatioData current amount is higher than its mean , then it is a good time to open a Sell Short trade which is to sell A and buy B . By Selling Short we are predicting RatioData decrease and have placed our trades accordingly . It is also a Buy Long entry signal if RatioData current amount is lower than its mean . In this circumstance , we can predict RatioData increase and open positions accordingly by buying A and selling B.
PAIR TRADE JUSTIFICTION


Lets now walk through choosing an entry signal . We will find all RatioData extremums
ALL EX
Then classify them by [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) 
AFTER KMEAN
Then we remove those calculated [centroids](https://en.wikipedia.org/wiki/Centroid) which are too close to the mean. 
AFTER REMOVAL

After finding all eligible centroids , we will signal for opening a position if RatioData current price is inside any of these extremums neighborhood.

SIGNAL CHECK . BOTH entry and non entry

Now that we have defined our way of considering a price as suitable for opening positions , we will investigate situations by which our opened positions must get closed .
 as our strategy suggests, we will close positions as wins when RatioData price is inside a neighborhood of its mean or even better, when they have passed the mean toward the profitable direction .
 
 PIC FOR BOTH CONITION
 condition 2vom bayad in beshe ke age az mean dar jahate dorost rad karde bud . na hatman ndazeye stoploss ta
 
 
 
 In other hand , our positions would lose if RatioData price moves opposite to what we have predicted to a amount that is higher than imposed Stop Loss. 
 
LOSE POS
 
 I have finished almost all technical aspects to this trading project and what has left is parameter tuning using 1 Minute data from 2008 to 2023 .
 These parameters consist of Window Frame , StopLoss Coefficient , Centroid Eligibility Coefficient , Centroid Neighborhood Coefficient , Mean Neighborhood Coefficient an Instruments Cool Down time .

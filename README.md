
This is a summery of my pair trade strategy . [Pair trade](https://en.wikipedia.org/wiki/Pairs_trade) is a [mean reversion](https://en.wikipedia.org/wiki/Mean_reversion_(finance)) strategy where our highest profits are when price is as far as possible from mean , and since we expect price`s inevitable return toward mean, we can make profit if we have bought or sold accordingly . Here is the pipe line :
![image](https://user-images.githubusercontent.com/76734519/228854482-d3f36b2b-c828-4696-ad3d-45792192a082.png)



We will investigate among 18 Forex pairs and update to the best possible pair at each time unit .We will pick a window amount slice of each instrument for every time unit and select(if exist)  the most two [cointegrated pairs by Engle-Granger Two-Step Method](https://corporatefinanceinstitute.com/resources/data-science/cointegration/) . Lets call them pair A and pair B .  As they are cointegrated , their ratio which is RatioData = A/B is changing around a constant amount which is its mean . Below heatmap is illustrating cointegration between instruments for a specific time interval with green color and how much each pair is cointegrated with others.
![HeatMap](https://user-images.githubusercontent.com/76734519/228843179-c1a01357-619f-44c0-bb11-21d2985603b2.png)



Now that we have our RatioData we will open and close positions looking at it . Our strategy is a mean reversion , therefore we will profit more if we open our trade when we are farthest from the mean . For instance , below image has marked some points where there is a considerble distance between them and mean . 
![image](https://user-images.githubusercontent.com/76734519/228852137-ad942223-82d6-4d6e-bfa6-b7f6fc238408.png)

When we sell short , we predict decrease for our data. Our data is A/B and if we predict its decrease then we need to open a position for decrease in A which is RatioData numerator and increase in B which is RatioData denominator . Therefore , we need to sell A and buy B . Buy long is the exact opposit. our data is far from and below the mean . So we predict its increase . For trading accordingly , we need to open positions that would profit from RatioData increase By buying A and selling B , we predecit increase for A/B . 


Lets now walk through choosing these entry s . First we need to find all RatioData extremums
![AllExtremum](https://user-images.githubusercontent.com/76734519/228848696-7b62408c-10fe-4723-a3d2-575c736100f7.png)

Then classify them by [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) 
![After-KMean](https://user-images.githubusercontent.com/76734519/228848755-168d3d82-d71d-486d-8565-dab1225aaecf.png)

Then we remove those calculated [centroids](https://en.wikipedia.org/wiki/Centroid) which are too close to the mean. 
![RemovingPoorCentroids](https://user-images.githubusercontent.com/76734519/228848868-317c3575-ca42-4053-a059-4c8a0736ac22.png)


After finding all eligible centroids , we will signal for opening a position if RatioData current price is inside any of these eligibles extremums neighborhood.

![image](https://user-images.githubusercontent.com/76734519/228855041-1104a37e-de9e-41d5-8e41-b06d41c6ed6b.png)

Now that we have defined our way of considering a price as suitable for opening positions , we will investigate situations by which our opened positions must get closed .As our strategy suggests, we will close positions as wins when RatioData price is inside a neighborhood of its mean or even better, when they have passed the mean toward the profitable direction .
 
Win if price in inside mean neighborhood
![image](https://user-images.githubusercontent.com/76734519/228859737-22070e3b-5768-4e60-8bc8-a7a687d66376.png)

Win if price has passed the mean toward profitable direction
 
 In other hand , our positions would lose if RatioData price moves opposite to what we have predicted to a amount that is higher than imposed Stop Loss. 
 
LOSE POS
 
 I have finished almost all technical aspects to this trading project and what has left is parameter tuning using 1 Minute data from 2008 to 2023 .In fact this pair trade can be performed on various time frames.
 Its parameters consist of Window Frame , StopLoss Coefficient , Centroid Eligibility Coefficient , Centroid Neighborhood Coefficient , Mean Neighborhood Coefficient an Instruments Cool Down time .

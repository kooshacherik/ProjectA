# Pair Trade
This is a summery of my pair trade strategy. [Pair Trade](https://en.wikipedia.org/wiki/Pairs_trade) is a [Mean Reversion](https://en.wikipedia.org/wiki/Mean_reversion_(finance)) strategy which benefits base on our prediction for prices to move toward a constant amount. Here is the **pipe line**:

![image](https://user-images.githubusercontent.com/76734519/228919768-c2632144-2da3-4652-aa5a-8f128d4dfd66.png)



We will investigate among 18 Forex pairs and update to the best possible pair at each time unit. We will pick a window (100 in images) amount slice of each instrument for every time unit and select the most two [cointegrated pairs by Engle-Granger Two-Step Method](https://corporatefinanceinstitute.com/resources/data-science/cointegration/). Lets call them pair A and pair B.  As they are cointegrated, their ratio which is ***RatioData = A/B*** is changing around a constant amount which is its mean. In below **heatmap**, cointegration between pairs has been illustrated with green color and their number being closer to 0:

![HeatMap](https://user-images.githubusercontent.com/76734519/228843179-c1a01357-619f-44c0-bb11-21d2985603b2.png)



Now that we have our ***RatioData***, we will open and close positions base on it. Our strategy is a mean reversion, therefore we will profit more if we open our trade positions when we are as far as possible from the mean. In below image, some points has been marked where there is a considerble distance between them and mean and therefore, they are **proper candidates** for opening positions. 
![image](https://user-images.githubusercontent.com/76734519/228869310-ccf78d12-f8c3-4a17-afbb-b07013d16666.png)



When we sell short, we predict decrease for our data. As our data is ***A/B*** and if we predict its decrease, then we need to open two positions,  one for decrease in ***A*** which is RatioData numerator and other one for increase in ***B*** which is RatioData denominator. Therefore, we need to sell ***A*** and buy ***B***. Buy long is the exact opposit. our data is still far from but, below the mean. So we predict its increase. For trading accordingly, we need to open positions that would profit from ***RatioData*** increase and predict for ***A/B*** raise, by buying ***A*** and selling ***B***.



Lets now walk through choosing these entries. First we find all ***RatioData*** **extremums**:

![AllExtremum](https://user-images.githubusercontent.com/76734519/228848696-7b62408c-10fe-4723-a3d2-575c736100f7.png)



Then classify extremums by **[K-Means](https://en.wikipedia.org/wiki/K-means_clustering)**:

![After-KMean](https://user-images.githubusercontent.com/76734519/228848755-168d3d82-d71d-486d-8565-dab1225aaecf.png)



Then we remove those calculated **[centroids](https://en.wikipedia.org/wiki/Centroid)** which are too close to the mean:

![image](https://user-images.githubusercontent.com/76734519/228866443-6f9e39b7-310a-48b0-bf0f-e779ada91b18.png)



After finding all eligible centroids, we will **singal to enter** to a position if ***RatioData*** current price is inside the neighborhood of any eligible centroid:

![image](https://user-images.githubusercontent.com/76734519/228946062-b3d772c6-9051-480a-8f79-cea94f5da2bf.png)



Now that we have defined our way of considering a price as suitable for opening positions, we will investigate situations by which our opened positions must get closed. As our strategy suggests, we will close positions as **win** when ***RatioData*** price is inside a neighborhood of its mean or even better, when they have passed the mean toward the profitable direction.
 
Win if price in inside mean neighborhood:

![image](https://user-images.githubusercontent.com/76734519/228862415-76451384-76c7-4353-b597-3cf2f678f724.png)


Also Win if price has passed the mean toward profitable direction:

![image](https://user-images.githubusercontent.com/76734519/228919165-ce7eaa3b-6c8d-4ed0-b5a8-ab92c73ed4d4.png)


In other hand, our positions would **lose** if ***RatioData*** price moves opposite to what we have predicted to a amount to a  higher amount than the imposed Stop Loss:
![image](https://user-images.githubusercontent.com/76734519/228864568-cfb5214f-b9e0-4c61-9275-013426ce55da.png)



What has left is **parameter tuning** and **backtesting** using 1-Minute data from 2008 to 2023. I believe this pair trade can be successfully deployed on various time frames as well. These parameters are  Window Frame length, StopLoss Coefficient, Centroid Eligibility Coefficient, Centroid Neighborhood Coefficient, Mean Neighborhood Coefficient and Instruments Cool Down time.

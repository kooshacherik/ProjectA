
This is a summery of my pair trade strategy . [Pair trade](https://en.wikipedia.org/wiki/Pairs_trade) is a [mean reversion](https://en.wikipedia.org/wiki/Mean_reversion_(finance)) strategy where our highest profits are when price is as far as possible from mean , and since we expect price`s inevitable return toward mean, we can make profit if we have bought or sold accordingly . We will investigate among 18 Forex pairs and update to the best possible pair at each time unit .We will pick a window amount slice of each instrument for every time unit and select(if exist)  the most two [cointegrated pairs by Engle-Granger Two-Step Method](https://corporatefinanceinstitute.com/resources/data-science/cointegration/) . Lets call them pair A and pair B .  As they are cointegrated , their ratio which is D = A/B is changing around a constant amount which is its mean .![Untitled](https://user-images.githubusercontent.com/76734519/228634689-a7763262-4184-4191-8cfe-0ba0fb12e669.png)![s](https://user-images.githubusercontent.com/76734519/228635003-542c8c8f-8e63-4544-b8bf-a77c07cb52b3.png)





We will open and close positions by analyzing D . When D current amount is higher than its mean , then it is a good time to open a Sell Short trade which is to sell A and buy B . By Selling Short we are predicting Ds decrease and have placed our trades accordingly . It is also a Buy Long entry signal if Ds current amount is lower than its mean . In this circumstance , we can predict Ds increase and open positions accordingly by buying A and selling B.

Lets walk through choosing an entry signal . We will find all Ds extremums and classify them by [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) and then remove calculated [centroids](https://en.wikipedia.org/wiki/Centroid) that are too close to the mean . After finding all eligible centroids , we will signal for opening a position if Ds current price is inside any of these extremums neighborhood.

Now that we have defined our way of considering a price as suitable for opening positions , we will investigate situations by which our opened positions must get closed .
 as our strategy suggests, we will close positions as wins when Ds price is inside a neighborhood of its mean or even better, when they have passed the mean toward the profitable direction . In other hand , our positions would lose if Ds price moves opposit to what we have predicted to a amount that is higher than imposed Stop Loss. 
 
 
 
 I have finished almost all technical aspects to this trading project and what has left is this project`s parameter tuning using 1 Minute data from 2008 to 2023 .
 These parameters consist of Window Frame , StopLoss Coefficient , Centroid Eligibility Coefficient , Centroid Neighborhood Coefficient , Mean Neighborhood Coefficient an Instruments Cool Down time .

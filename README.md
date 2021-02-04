# ScalpingAlgoTrader

The goal of this project is to create a profitable open-source algorithmic trading strategy. The current version lays the groundwork for that vision. 


## About

ScalpingAlgoTrader uses standard deep neural networks (DNN) to predict the next-minute price direction of a stock, i.e. to predict if the stock price will increase or decrease in 60 seconds. The DNNs are trained daily on Wharton's millisecond ticker data. Each DNN achieves ~65% accuracy on the test set with a typical variance of about 3% from the validation set and 5% from the test set. However, accuracy can be as low as 52% and as high as 75%. Recurrent Neural Networks (RNN) were used for a time, but they were, surprisingly, less effective than DNNs. 

The idea of using DNNs over RNNs was inspired by [Arevalo et al.](https://www.researchgate.net/publication/305214717_High-Frequency_Trading_Strategy_Based_on_Deep_Neural_Networks) and the use of Keras was inspired by [Paspanthong et al.](http://cs229.stanford.edu/proj2019spr/report/28.pdf)

The trading strategy uses the Alpaca API to make commissionless trades and uses a 2:5 risk/reward ratio. Unfortunately, this version of the algorithm is not profitable. There are several reasons:
  1. Execution times with Alpaca are too slow. Many times after submitting an order, the majority of a desirable movement will have already     occurred before the order is filled, and we are left in an undesirable position. Moving to a commissioned brokerage with faster execution would fix this.
  2. Risk/reward ratio. The ideal risk/reward ratio should be calibrated (maybe using machine learning methods?) using historical data for each stock that is traded. The current algorithm uses a static risk reward ratio, meaning that many potentially good positions are exited too early (for a loss), and also that many green positions turn red before exit because the static reward percentage isn't reached.
  3. Not taking into account news, general market trends, sector sympathy, etc. It doesn't matter how well a DNN can predict yesterday's price movements if news comes out during premarket that affects the entire trading day. Another script should be written to tune into sources like Benzinga, Finviz, and SEC filings in order to add weight to the live fundamentals of a stock in price predictions.
  4. The current algorithm only trades a single stock. This means that nothing is diversified, and risk isn't balanced by portfolio.
  5. The (scalping) timeframe is usually too small for desirable movements. Since this DNN uses binary classification, every movement in the desirable price-direction is classified the same, so it doesn't account for very small movements that expose the account to large risk for too small of a reward.
 
I am currently working on addressing these issues using [Machine Learning for Algorithmic Trading](https://github.com/stefan-jansen/machine-learning-for-trading) by Stefan Jansen.

# Usage

## Retrieving and wrangling Wharton data
In order to gather this data, you must have a Wharton Research Data Services (WDRS) account or be able to retrieve a WDRS day pass through your institution. Once you are connected, click on TAQ --> Consolidated Trades. Then you have to download three sets of data:
1. The training set: Allocate two months of ticker data for the training set. Choose your stock. Make sure to choose 9:30 - 16:00 for the time period. Also make sure you select the Price of Trade and Volume of Trade. Download in CSV file. Rename CSV file train_set and place in CleanData folder.
2. The validation set: The next month of data after the training set data. Same criteria. Rename CSV file validation_set and place in CleanData folder.
3. The test set: The next month of data after the. Rename CSV file test_set and place in CleanData folder.

Run 
```shell
$ python3 utils.py
```
to process the individual trade data into minute data. You may find that some of the trades downloaded from WDRS have empty data. You will have to delete these empty trades from the CSV before you run utils.py.

## Training and hypertuning DNN
Next, you will have to set up a GCP Ubuntu 18.04 instance and upload the code to a bucket. Instructions on how to do that [here](https://medium.com/automation-generation/build-a-day-trading-algorithm-and-run-it-in-the-cloud-for-free-805450150668). Once everythng is uploaded, run 
```shell
$ python3 main.py
```
This script will simultaneously train and tune the DNN. It will output the finished DNN to a folder called saved_model.

## Trading Strategy
Now you should make an account with [Alpaca](https://alpaca.markets/). Update the auth data in the main method of algo.py with your credentials. Now all you have to do is run 
```shell
$ python3 algo.py
```
on GCP, and the algo will start trading.

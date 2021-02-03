# ScalpingAlgoTrader

The goal of this project is to open-source a profitable algorithmic trading strategy. The current version lays the groundwork for that vision. 


## About

ScalpingAlgoTrader uses standard deep neural networks (DNN) to predict the next-minute price direction of a stock, i.e. to predict if the stock price will increase or decrease in 60 seconds. The DNNs are trained daily on Wharton's millisecond ticker data. Each DNN achieves ~65% accuracy on the test set with a typical variance of about 3% from the validation set and 5% from the test set. However, accuracy can be as low as 52% and as high as 75%. Recurrent Neural Networks (RNN) were used for a time, but they were, surprisingly, less effective than DNNs. 

The idea of using DNNs over RNNs was inspired by [Arevalo et al.] (https://www.researchgate.net/publication/305214717_High-Frequency_Trading_Strategy_Based_on_Deep_Neural_Networks) and the use of Keras was inspired by <a href="http://cs229.stanford.edu/proj2019spr/report/28.pdf" Paspanthong et al. </a>.

The trading strategy uses the Alpaca API to make commissionless trades and uses a 2:5 risk/reward ratio. Unfortunately, this version of the algorithm is not profitable. There are several reasons:
  1. Execution times with Alpaca are too slow. Many times after submitting an order, the majority of a desirable movement will have already     occurred before the order is filled, and we are left in an undesirable position. Moving to a commissioned brokerage with faster execution would fix this.
  2. Risk/reward ratio. The ideal risk/reward ratio should be calibrated (maybe using machine learning methods?) for each stock that is traded using historical data. The current algorithm uses a static risk reward ratio, meaning that many potentially good positions are exited too early (for a loss), and also that many green positions turn red because the static reward percentage isn't reached.
  3. Not taking into account news, general market trends, sector sympathy, etc. It doesn't matter how well a DNN can predict yesterday's price movements if news comes out during premarket that affects the entire trading day. Another script should be written to tune into sources like Benzinga, Finviz, and SEC filings in order to add weight to the fundamentals of a stock in price predictions.
  4. The current algorithm only trades a single stock. This means that nothing is diversified, and risk isn't balanced by portfolio
  5. The timeframe is usually too small for desirable movements. Since this DNN uses binary classification, every movement in the desirable price-direction is classified the same, so it doesn't account for very small movements that expose the account to large risk for too small of a reward.
 
I am currently working on addressing these issues, specifically using <a href="https://github.com/stefan-jansen/machine-learning-for-trading" Machine Learning for Algorithmic Trading by Stefan Jansen</a>.

# Usage

## Retrieving and wrangling Wharton data
Getting the wharton data
fixing NONE issues in CSV files

## Training and hypertuning DNN
running utils.py
Running main.py, training and hypertuning in GCP 

## Trading Strategy
Setting up alpaca 
Running algo.py

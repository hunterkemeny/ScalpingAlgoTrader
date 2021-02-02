import asyncio
import websockets
import alpaca_trade_api as tradeapi
import json
import logging
import datetime as dt
import requests
import asyncio
import pandas as pd
import sys
import time
import numpy as np
import IPython
import kerastuner as kt
import tensorflow as tf
import math
from utils import get_clean_data, get_final_clean_data, get_test_data
from linreg import estimate_coef
from DNN import RNNModel, ClearTrainingOutput, model_builder
from main import get_model, create_features
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras import optimizers, utils

async def on_open():
    print("opened")
    async with websockets.connect(socket, ping_interval=None) as websocket:
        # Authorize user.
        await websocket.send(json.dumps(auth_data))
        authorization = await websocket.recv()
        print(authorization)
        # Connect to the AAPL trade stream.
        listen_message = {"action": "listen", "data": {"streams": ["T.AAPL"]}}
        listening = await websocket.send(json.dumps(listen_message))
        await on_message(websocket)
        print(listening)

async def on_message(websocket):
    # This method is called every time a trade occurs.
    async for message in websocket:
        # Don't process the message if it is the connection message.
        if message[11] == 'l':
            continue
        else:
            print()
            # Determine if a new minute has started by comparing the last minute stored in trade_minutes to the current minute.
            if get_current_minute() != trade_minutes[-1]:
                new_minute[0] = True
            orders = get_orders()
            # Use the order list to determine if we currently hold any positions.
            if orders == []: current_operation[0] = "NONE"
            # Make predictions and trades at the half minute because start-of-the-minute trading volume is too volatile, which causes pricing inefficiencies.
            if new_minute[0] == True and get_current_second() >= 30:
                new_minute[0] = False
                minute_trade_prices.append(get_price(message)) 
                # The trade_seconds list is used for linear regression; since this algo trades on the half-minute, 
                # the second values have to be shifted for each half-minute so that they appear in the correct order.
                if get_current_second() < 30: trade_seconds.append(get_current_second() + 30)
                else: trade_seconds.append(get_current_second() - 30)
                trade_minutes.clear()
                trade_minutes.append(get_current_minute())
                main()
            else:
                # If a message comes in and it is not a new minute, update current_price, trade_seconds, and minute_trade_prices.
                print("current second: ", get_current_second())
                minute_trade_prices.append(get_price(message)) 
                if get_current_second() < 30: trade_seconds.append(get_current_second() + 30)
                else: trade_seconds.append(get_current_second() - 30)
                trade_minutes.append(get_current_minute())
                print(minute_trade_prices)
                current_price = minute_trade_prices[-1]
                print("current operation: ", current_operation[0])
                print("current price: ", current_price)
                print("last transaction price: ", tracker_dict['Last Transaction Price'][-1])
                
                # If the price has increased by at least 5% in the long position since the last transaction, sell and take profit.
                if current_operation[0] == "LONG" and current_price >= tracker_dict['Last Transaction Price'][-1]*1.05:
                    cancel_stop_loss()
                    sell()
                    transaction_price = get_transaction_price()
                    tracker_dict['Last Transaction Price'].append(transaction_price)
                    current_operation[0] == "NONE"
                # If the price has decreased by at least 5% in the short position since the last transaction, cover and take profit.
                if current_operation[0] == "SHORT" and current_price <= tracker_dict['Last Transaction Price'][-1]*0.97:
                   cancel_stop_loss()
                   buy()
                   transaction_price = get_transaction_price()
                   tracker_dict['Last Transaction Price'].append(transaction_price)
                   current_operation[0] == "NONE"
def main():
    # Collect and preprocess data so that it is consistent with the format found in main.py.
    average_price = get_average_price()
    standard_deviation = get_standard_deviation(average_price)
    pseudo_log_return = math.log(average_price) - math.log(tracker_dict['Average Price'][-1])
    tracker_dict['Hour'].append(get_current_hour())
    tracker_dict['Minute'].append(get_current_minute())
    tracker_dict['Average Price'].append(average_price)
    tracker_dict['Standard Deviation'].append(standard_deviation)
    tracker_dict['Pseudo-Log-Return'].append(pseudo_log_return)
    tracker_dict['Trend Indicator'].append(estimate_coef(trade_seconds[:-1], minute_trade_prices[:-1])[1])
    print(tracker_dict)

    features = [[tracker_dict['Hour'][-3], tracker_dict['Minute'][-3], tracker_dict['Average Price'][-3], tracker_dict['Standard Deviation'][-3], tracker_dict['Pseudo-Log-Return'][-3]], [tracker_dict['Hour'][-2], tracker_dict['Minute'][-2], tracker_dict['Average Price'][-2], tracker_dict['Standard Deviation'][-2], tracker_dict['Pseudo-Log-Return'][-2]], [tracker_dict['Hour'][-1], tracker_dict['Minute'][-1], tracker_dict['Average Price'][-1], tracker_dict['Standard Deviation'][-1], tracker_dict['Pseudo-Log-Return'][-1]]]
    features_df = pd.DataFrame({
        'Hour': [tracker_dict['Hour'][-3], tracker_dict['Hour'][-2], tracker_dict['Hour'][-1]],
        'Minute': [tracker_dict['Minute'][-3], tracker_dict['Minute'][-2], tracker_dict['Minute'][-1]],
        'Pseudo-Log-Return': [tracker_dict['Pseudo-Log-Return'][-3], tracker_dict['Pseudo-Log-Return'][-2], tracker_dict['Pseudo-Log-Return'][-1]],
        'Standard Deviation': [tracker_dict['Standard Deviation'][-3], tracker_dict['Standard Deviation'][-2], tracker_dict['Standard Deviation'][-1]],
        'Trend Indicator': [tracker_dict['Trend Indicator'][-3], tracker_dict['Trend Indicator'][-2], tracker_dict['Trend Indicator'][-1]]
    })
    
    features = features_df[headers].values
    print(features)
    Xtest_tmp = np.append(Xtest, features, axis=0)
    scale = MinMaxScaler(feature_range=(0, 1))
    Xtest_tmp = scale.fit_transform(Xtest_tmp)
    print(Xtest_tmp)
    features = Xtest_tmp[-3:]
    print(features)
    features = create_features(features, 2)
    print(features)

# Track the accuracy of the predictions made by the neural network and report them.
    if len(predictions) > 0:
        if minute_trade_prices[-1] <= minute_trade_prices[0] and predictions[-1] == True:
            correct_predictions.append(1)
            print("Accuracy: ", sum(correct_predictions)/len(correct_predictions))
        elif minute_trade_prices[-1] >= minute_trade_prices[0] and predictions[-1] == False:
            correct_predictions.append(1)
            print("Accuracy: ", sum(correct_predictions)/len(correct_predictions))
        else:
            correct_predictions.append(0)
            print("Accuracy: ", sum(correct_predictions)/len(correct_predictions))
    
    # Make the prediction based on the data collected and computed above.
    will_decrease = predict_decrease(features)
    predictions.append(will_decrease)

    print(will_decrease)
    # Use 3:5 Risk Reward ratio.
    if will_decrease and current_operation[0] == "NONE":
        # If we predict that the price is going to decrease, and we currently don't hold a position, then we want to short.
        sell()
        # It takes about 0.5 seconds for the transaction to process, so we have to wait to retrieve the transaction price.
        time.sleep(0.5)
        transaction_price = get_transaction_price()
        tracker_dict['Last Transaction Price'].append(transaction_price)
        # Set a stop loss for if the prediction is incorrect.
        stop_limit_buy(transaction_price*1.03, transaction_price*1.035)
        current_operation[0] = "SHORT"
    elif not will_decrease and current_operation[0] == "NONE":
        # If we predict that the price is going to increase, and we currently don't hold a position, then we want to long.
        buy()
        time.sleep(0.5)
        transaction_price = get_transaction_price()
        tracker_dict['Last Transaction Price'].append(transaction_price)
        stop_limit_sell(transaction_price*0.97, transaction_price*0.965)
        current_operation[0] = "LONG"
    elif not will_decrease and current_operation[0] == "SHORT":
        # If we predict that the price is going to increase, and we are currently in a short position, 
        # then we want to cover the short and flip positions.
        cancel_stop_loss()
        buy_short()
        time.sleep(0.5)
        buy()
        time.sleep(0.5)
        transaction_price = get_transaction_price()
        tracker_dict['Last Transaction Price'].append(transaction_price)
        stop_limit_sell(transaction_price*0.97, transaction_price*0.965)
        current_operation[0] = "LONG"
    elif will_decrease and current_operation[0] == "LONG":
        # If we predict that the price is going to decrease, and we are currently in a short position, 
        # then we want to exit the position and flip to short.
        cancel_stop_loss()
        sell_long()
        time.sleep(0.5)
        sell()
        time.sleep(0.5)
        transaction_price = get_transaction_price()
        tracker_dict['Last Transaction Price'].append(transaction_price)
        stop_limit_buy(transaction_price*1.03, transaction_price*1.035)
        current_operation[0] = "SHORT"

    minute_trade_prices.clear()
    trade_seconds.clear()

def get_orders():
    orders_endpoint = "https://paper-api.alpaca.markets/v2/orders"
    orders_api = tradeapi.REST(API_KEY, SECRET_KEY, base_url=orders_endpoint)
    r = requests.get(orders_endpoint, headers=HEADERS)
    response = json.loads(r.content)
    return response

def is_stop_limit():
    if get_orders()[0]['order_type'] == 'stop_limit': return True
    else: return False

def get_price(message):
    price_index = message.index("p")
    price_string = message[price_index:]
    comma_index = price_string.index(",")
    price = float(price_string[3:comma_index])
    return price

def get_transaction_price():
    activities_endpoint = "https://paper-api.alpaca.markets/v2/account/activities/FILL"
    r = requests.get(activities_endpoint, headers=HEADERS)
    response = json.loads(r.content)
    return float(response[0]['price'])

def buy(profit_price=0, loss_price=0):
    qty = math.floor(float(get_buying_power())/float(minute_trade_prices[-1]))
    data = {
        "symbol": "AAPL",
        "qty": qty,
        "side": "buy",
        "type": "market",
        "time_in_force": 'gtc'
    }
    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
    response = json.loads(r.content)
    print("BUY RESPONSE")
    print(response)
    order_id = response['id']
    stop_order_id[0] = order_id
    current_qty[0] = qty

def buy_short():
    # This is the function used to cover the short position.
    qty = current_qty[0]
    data = {
        "symbol": "AAPL",
        "qty": current_qty[0],
        "side": "buy",
        "type": "market",
        "time_in_force": 'gtc'
    }
    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
    response = json.loads(r.content)
    print("BUY SHORT RESPONSE")
    print(response)

def stop_limit_buy(stop_price, limit_price):
    data = {
        "symbol": "AAPL",
        "qty": current_qty[0],
        "side": "buy",
        "type": "stop_limit",
        "stop_price": stop_price,
        "limit_price": limit_price,
        "time_in_force": 'gtc'
    }
    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
    response = json.loads(r.content)
    print("BUY STOP LIMITS RESPONSE")
    print(response)
    order_id = response['id']
    stop_order_id[0] = order_id

def sell(profit_price=0, loss_price=0):
    qty = math.floor(float(get_buying_power())/float(minute_trade_prices[-1]))
    data = {
        "symbol": "AAPL",
        "qty": qty,
        "side": "sell",
        "type": "market",
        "time_in_force": 'gtc'
    }
    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
    response = json.loads(r.content)
    print("SELL RESPONSE")
    print(response)
    order_id = response['id']
    stop_order_id[0] = order_id
    current_qty[0] = qty

def sell_long():
    # This is the function used to exit the long position.
    qty = current_qty[0]
    data = {
        "symbol": "AAPL",
        "qty": current_qty[0],
        "side": "sell",
        "type": "market",
        "time_in_force": 'gtc'
    }
    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
    response = json.loads(r.content)
    print("SELL LONG RESPONSE")
    print(response)

def stop_limit_sell(stop_price, limit_price):
    data = {
        "symbol": "AAPL",
        "qty": current_qty[0],
        "side": "sell",
        "type": "stop_limit",
        "stop_price": stop_price,
        "limit_price": limit_price,
        "time_in_force": 'gtc'
    }
    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
    response = json.loads(r.content)
    print("SELL STOP LIMITS RESPONSE")
    print(response)
    order_id = response['id']
    stop_order_id[0] = order_id

def replace_stop_loss(stop_price, limit_price):
    stop_loss_id = stop_order_id[0]
    order_endpoint = "https://paper-api.alpaca.markets/v2/orders/{}".format(stop_loss_id)
    data = {
        "qty": current_qty[0],
        "limit_price": limit_price,
        "stop_price": stop_price
    }
    r = requests.patch(order_endpoint, json=data, headers=HEADERS)
    response = json.loads(r.content)
    order_id = response['id']
    stop_order_id[0] = order_id

def cancel_stop_loss():
    stop_loss_id = stop_order_id[0]
    order_endpoint = "https://paper-api.alpaca.markets/v2/orders/{}".format(stop_loss_id)
    r = requests.delete(order_endpoint, headers=HEADERS)

def get_buying_power():
    r = requests.get(ACCOUNT_URL, headers=HEADERS)
    response = json.loads(r.content)
    cash = response['equity']
    return cash

def get_current_hour():
    return dt.datetime.now().hour

def get_current_minute():
    return dt.datetime.now().minute

def get_current_second():
    return dt.datetime.now().second

def get_average_price():
    return sum(minute_trade_prices)/len(minute_trade_prices)

def get_standard_deviation(average_price):
    sum_transaction_price_minus_average_price_squared = 0
    for price in minute_trade_prices:
        sum_transaction_price_minus_average_price_squared += (price - average_price)**2
    return math.sqrt(sum_transaction_price_minus_average_price_squared/len(minute_trade_prices))

def get_pseudo_log_return(current_price, last_avg_price):
    return math.log(current_price) - math.log(last_avg_price)

def predict_decrease(features):
    print(features)
    prediction = model.predict(features)
    print(prediction)
    if prediction >= 0.5: return True
    else: return False

if __name__ == "__main__":
    # Authentication and endpoint setup.
    paper_endpoint = "https://paper-api.alpaca.markets"
    API_KEY = "PKLWHJJW5TYFJ630YZPR"
    SECRET_KEY = "h32ry1ZpwdagubrbvZsWVbDDJIaNMnpUchu3WkH2"
    auth_data = {"action": "authenticate", "data": {"key_id": API_KEY, "secret_key": SECRET_KEY}}
    ORDERS_URL = "{}/v2/orders".format(paper_endpoint)
    ACCOUNT_URL = "{}/v2/account".format(paper_endpoint)
    HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}
    socket = "wss://data.alpaca.markets/stream"
    api = tradeapi.REST(API_KEY, SECRET_KEY, base_url=paper_endpoint)

    # Global variables are used because it is impossible to pass variables into the asyncio methods. However, care is taken to use 
    # Python lists rather than primitives, keeping in mind the environment-diagram hierarchy of items stored in lists. This allows
    # each method to change single values in each list without having to worry about the usual problems associated with global variables.

    # Use minute_trade_prices to store the transaction prices of all trades in a given minute.
    minute_trade_prices = []
    # predictions stores every prediction made by the neural network.
    predictions = []
    # correct_predictions is used to track the accuracy of the predictions.
    correct_predictions = []
    # trade_minutes is used to determine when a new minute has started, causing the neural network to make a prediction.
    trade_minutes = [get_current_minute()]
    # The intraminute second for each trade is logged in trade_seconds so that linear regression can be performed on the intraminute trade prices.
    trade_seconds = []
    # tracker_dict collects the relevant data for each minute. The first values are hard-coded for the look_back window, so the first three minutes are inaccurate.
    tracker_dict = {'Hour': [0, 0, 0], 'Minute': [0, 0, 0], 'Average Price': [1, 1, 1], 'Standard Deviation': [0, 0, 0], 'Pseudo-Log-Return': [0, 0, 0], 'Trend Indicator': [0, 0, 0], 'Last Transaction Price': [0, 0, 0]}
    current_operation = ["NONE"]
    stop_order_id = [""]
    current_qty = [""]
    new_minute = [True]

    # Collect the data that will be used to normalize the feature vector.
    headers = ['Pseudo-Log-Return', 'Trend Indicator']
    df_test = get_test_data()
    Xtest = df_test[headers].values 
    # Load the saved model.
    model = tf.keras.models.load_model('./saved_model/my_model')
    
    asyncio.get_event_loop().run_until_complete(on_open())

import pandas as pd 
import csv
from math import log, sqrt

def create_set(input_file, output_file):
    """
    Create a CSV file with the relevant data needed to predict the next-minute direction of stock prices.

    Inputs:
        input_file: A CSV file from the Wharton TAQ database containing the price, time, and trade volume 
                    of all the transactions of a particular set of stocks during a given time period.
        output_file: The string of the file to export the new CSV.
    """
    data_dict = {'Date': [], 'Hour': [], 'Minute': [], 'Average Price': [], 'Pseudo-Log-Return': [], 'Direction': [], 'Standard Deviation': [], 'Trend Indicator': []}
    input_df = pd.read_csv(input_file)

    start_date, end_date = get_start_end_dates(input_df)
    dates = get_dates(input_df, start_date)

    data_dict = insert_dates(dates, data_dict)
    data_dict = insert_average_price(input_df, data_dict)
    data_dict = insert_directional_classification(data_dict)
    data_dict = insert_pseudo_log_return(data_dict)
    data_dict = insert_standard_deviation(input_df, data_dict)
    data_dict = insert_trend_indicator(input_df, data_dict)

    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(f'{output_file}')

def get_start_end_dates(input_df):
    return input_df['DATE'][0], input_df['DATE'].values[-1]

def get_dates(input_df, start_date):
    """
    Get the open dates contained in input_df.

    Inputs:
        input_df: A pandas DataFrame corresponding to a particular CSV file.
        start_date: A string representing the first date of input_df.
    Return:
        dates: A list of the open dates contained in input_df.
    """
    dates = []
    current_date = start_date
    dates.append(current_date)
    for index, value in input_df['DATE'].items():
        if value == current_date: pass
        else:
            current_date = value
            dates.append(current_date)
    return dates

def insert_dates(dates, data_dict):
    """
    Insert the date, hour, and minute for each day in dates.

    Inputs:
        dates: A list of dates corresponding to trading days.
        data_dict: A dictionary containing the relevant data.
    Return:
        data_dict: Modified to include dates.
    """
    for date in dates:
        for hour in range(7):
            for minute in range(60):
                if hour == 0 and minute < 30:
                    pass
                else:
                    data_dict['Hour'].append(hour)
                    data_dict['Date'].append(date)
                    data_dict['Minute'].append(minute)
    return data_dict

def insert_average_price(input_df, data_dict):
    """
    Calculate and insert the one-minute average of transaction prices into data_dict.

    Inputs:
        input_df: A pandas DataFrame corresponding to a particular CSV file.
        data_dict: A dictionary containing the relevant data (the open minutes, in this case)
    Return:
        data_dict: Modified to include average price.
    """
    time_list = input_df['TIME_M']
    transaction = 0
    i = 0
    for minute in data_dict['Minute']:
        print('average price time: ', len(data_dict['Minute']) - i)
        i += 1
        sum_price = 0
        average_price = 0
        num_transactions = 0
        # Need try, except statement to account for the last minute, when get_row_minute throws an exception 
        # because row += 1 eventually produces an index bound error.
        try:
            # Get the transaction_price and transaction_time for each transaction during minute.
            while int(get_row_minute(transaction, time_list)) == minute:
                sum_price += input_df['PRICE'][transaction]
                num_transactions += 1
                transaction += 1
        except:
            pass
        # Calculate average_price.
        if num_transactions == 0:
            num_transactions = 1
            print('NUM TRANSACTIONS = 0')
        average_price = sum_price/num_transactions
        data_dict['Average Price'].append(average_price)
    return data_dict

def get_row_minute(transaction, time_list):
    """
    Inputs:
        time_list: a list of times at which corresponding transactions took place.
        transaction: an index corresponding to a particular transaction.
    Return:
        transaction_second: a string representing the second.milisecond that the transaction took place.
    """
    # Get the time corresponding to transaction.
    transaction_time = time_list[transaction]
    if transaction_time[0] != '1': row_minute = transaction_time[2:4]
    else: row_minute = transaction_time[3:5]
    return row_minute
    
# Classifies increase in price or staying the same as 0, decrease in price as 1
def insert_directional_classification(data_dict):
    """
    Insert the directional classification for each minute into data_dict. 1 classification if the next minute
    has a lesser average price; 0 classification if the next minute has a greater than or equal price.
    """
    for price_index in range(len(data_dict['Average Price'])):
        # Special case: if this is the last item in data_dict, append 0 because there is no later price to classify.
        if price_index >= len(data_dict['Average Price']) - 6: data_dict['Direction'].append(0)
        # elif abs(round(data_dict['Average Price'][price_index], 2) - round(data_dict['Average Price'][price_index + 1], 2)) < round(data_dict['Average Price'][price_index], 2)*buffer: data_dict['Direction'].append(0)
        elif round(data_dict['Average Price'][price_index], 2) <= round(data_dict['Average Price'][price_index + 5], 2): data_dict['Direction'].append(0)
        else: data_dict['Direction'].append(1)
    return data_dict

def insert_pseudo_log_return(data_dict):
    """
    Calculate and insert the one-minute pseudo-log-return of average prices for each minute into data_dict.
    pseudo-log-return = log(average_price_current_minute) - log(average_price_last_minute)

    Inputs:
        data_dict: A dictionary containing the relevant data (the average price, in this case)
    Return:
        data_dict: Modified to include pseudo-log-return.
    """
    for row in range(len(data_dict['Average Price'])):
        # Special case: for the first row, there is no previous price, so insert dummy 0 value.
        if row == 0:
            pseudo_log_return = 0
            data_dict['Pseudo-Log-Return'].append(pseudo_log_return)
        else:
            if data_dict['Average Price'][row] == 0 or data_dict['Average Price'][row - 1] == 0:
                pseudo_log_return = 1
                print('PSEDUO-LOG RETURN = 0')
                data_dict['Pseudo-Log-Return'].append(pseudo_log_return)
            else:
                pseudo_log_return = log(data_dict['Average Price'][row]) - log(data_dict['Average Price'][row - 1])
                data_dict['Pseudo-Log-Return'].append(pseudo_log_return)
    return data_dict

def insert_standard_deviation(input_df, data_dict):
    """
    Calculate and insert the one-minute standard deviation of transaction prices for each minute into data_dict.

    Inputs:
        input_df: A pandas DataFrame corresponding to a particular CSV file.
        data_dict: A dictionary containing the relevant data (the average price and open minutes, in this case)
    Return:
        data_dict: Modified to include standard deviation.
    """
    time_list = input_df['TIME_M']
    transaction = 0
    i = 0
    for minute_index in range(len(data_dict['Minute'])):
        print('standard deviation time: ', len(data_dict['Minute']) - i)
        i += 1
        minute = data_dict['Minute'][minute_index]
        average_price = data_dict['Average Price'][minute_index]
        sum_transaction_price_minus_average_price_squared = 0
        num_transactions = 0
        # Need try, except statement to account for the last minute, when get_row_minute throws an exception 
        # because row += 1 eventually produces an index bound error.
        try:
            # Calculate the sum_transaction_price_minus_average_price_squared and update num_transactions 
            # for each transaction during minute.
            while int(get_row_minute(transaction, time_list)) == minute:
                sum_transaction_price_minus_average_price_squared = (input_df['PRICE'][transaction] - average_price)**2
                num_transactions += 1
                transaction += 1
        except:
            pass
        if num_transactions == 0:
            num_transactions = 1
            print('NUM TRANSACTIONS = 0')
        # Calculate standard_deviation
        standard_deviation = sqrt((sum_transaction_price_minus_average_price_squared)/num_transactions)
        data_dict['Standard Deviation'].append(standard_deviation)
    return data_dict

def insert_trend_indicator(input_df, data_dict):
    """
    Calculate and insert the one-minute trend indicators–the slope of the equation for the best fit line obtained 
    by linear regression–for each minute into data_dict.

    Inputs:
        input_df: A pandas DataFrame corresponding to a particular CSV file.
        data_dict: A dictionary containing the relevant data (the open minutes, in this case)
    Return:
        data_dict: Modified to include trend indicators.
    """
    time_list = input_df['TIME_M']
    transaction = 0
    i = 0
    for minute in data_dict['Minute']:
        print('trend indicator time: ', len(data_dict['Minute']) - i)
        i += 1
        transaction_prices = []
        transaction_time = []
        # Need try, except statement to account for the last minute, when get_row_minute throws an exception 
        # because row += 1 eventually produces an index bound error.
        try:
            # Get the transaction_price and transaction_time for each transaction during minute.
            while int(get_row_minute(transaction, time_list)) == minute:
                transaction_prices.append(float(input_df['PRICE'][transaction]))
                transaction_time.append(float(get_row_second(time_list, transaction)))
                transaction += 1
        except:
            pass
        trend_indicator = estimate_coef(transaction_time, transaction_prices)[1]
        data_dict['Trend Indicator'].append(trend_indicator)
    return data_dict

# Geeks for Geeks https://www.geeksforgeeks.org/linear-regression-python-implementation/
def estimate_coef(x, y): 
    """
    Return the estimated coefficients of line of best fit given a vector of inputs and a corresponding vector
    of outputs.
    """
    try:
        # number of observations/points 
        x = np.array(x)
        y = np.array(y)
        n = np.size(x) 
        # mean of x and y vector 
        m_x, m_y = np.mean(x), np.mean(y) 
        # calculating cross-deviation and deviation about x 
        SS_xy = np.sum(y * x) - n * m_y * m_x 
        SS_xx = np.sum(x * x) - n * m_x * m_x 
        # calculating regression coefficients 
        b_1 = SS_xy / SS_xx 
        b_0 = m_y - b_1 * m_x 
        return (b_0, b_1) 
    except:
        return 0

def get_row_second(time_list, transaction):
    """
    Inputs:
        time_list: a list of times at which corresponding transactions took place.
        transaction: an index corresponding to a particular transaction.
    Return:
        transaction_second: a string representing the second.milisecond that the transaction took place.
    """
    # Get the time corresponding to transaction.
    transaction_time = time_list[transaction]
    if transaction_time[0] != '1': row_second = transaction_time[5:]
    else: row_second = transaction_time[6:]
    return row_second 

def get_test_data():
    """
    Return: 
        test: a set of data that is used to to normalize the live minute data in algo.py.
    """
    test = pd.read_csv('./CleanData/test.csv', error_bad_lines=False)
    return test

def get_clean_data():
    """
    Return the train, validation, and test set CSV files as pandas DataFrames.
    """
    train = pd.read_csv('./CleanData/train.csv', error_bad_lines=False)
    validation = pd.read_csv('./CleanData/validation.csv', error_bad_lines=False)
    test = pd.read_csv('./CleanData/test.csv', error_bad_lines=False)
    return train, validation, test

if __name__ == "__main__":
    create_set('./CleanData1/validation_set.csv', './CleanData/validation.csv')
    create_set('./CleanData1/train_set.csv', './CleanData/train.csv')
    create_set('./CleanData1/test_set.csv', './CleanData/test.csv')
    

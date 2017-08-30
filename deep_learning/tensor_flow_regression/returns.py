import pandas as pd
import numpy as np


def get_data():
    goog = 'data/GOOG.csv'
    sp_500 = 'data/^GSPC.csv'

    # convert the CSV files to frames of the data
    goog_df = pd.read_csv(goog, sep=',', usecols=[0,5], names=['Date', 'Goog'], header=0)
    sp_df = pd.read_csv(sp_500, sep=',', usecols=[0,5], names=['Date', 'SP500'], header=0)

    # add S&P 500 to our Google prices data frame
    goog_df['SP500'] = sp_df['SP500']

    # convert date values to Pandas datetime objects
    goog_df['Date'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

    # sort the data by date
    goog_df = goog_df.sort_values(['Date'], ascending=[True])

    # grab our returns and % delta
    returns = goog_df[[dtype for dtype in dict(goog_df.dtypes) if dict(goog_df.dtypes)[dtype] in ['float64', 'int64']]].pct_change()

    # convert our SP500 and Goog column series
    X = np.array(returns['SP500'])[1:]
    Y = np.array(returns['Goog'])[1:]

    return X, Y


def read_data_from_file(file_name):
    # specify the columns that we care about
    data = pd.read_csv(file_name, sep=',', usecols=[0,5], names=['Date', 'Price'], header=0)

    # convert date format
    data['Date'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

    # sort ascending
    data = data.sort_values(['Date'], ascending=[True])

    # get the percentage change
    returns = data[[dtype for dtype in dict(data.dtypes) if dict(data.dtypes)[dtype] in ['float64', 'int64']]].pct_change()

    # remove the first item before returning the ndarray
    return np.array(returns['Price'])[1:]


def get_nasdaq_oil_xom_data():
    nasdaq = read_data_from_file('data/^IXIC.csv')
    oil = read_data_from_file('data/USO.csv')
    xom = read_data_from_file('data/XOM.csv')

    return nasdaq, oil, xom

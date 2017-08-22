import numpy as np
import numpy as np
import csv

from sklearn.svm import SVR


def retrieve_data(file):
    '''
    Takes the name of a file and return a tuple containing a list of dates and a list of prices

    Args:
        file: The name of the file that we wish to extract data from
    Returns:
        A tuple containing a list of dates and a list of prices
    '''
    dates = []
    prices = []

    # open our .csv file and extract the dates and prices from it
    with open(file, 'r') as csv_file:
        file_reader = csv.reader(csv_file)
        next(file_reader) # iterate past the header line
        dates = [int(row[0].split('_')[0]) for row in file_reader]
        prices = [float(row[1]) for row in file_reader]
    return (dates, prices)


def forecast_prices(dates, prices):
    '''
    Takes a list of dates and a list of prices and returns a forecast

    Args:
        dates: a list of dates
        prices: a list of prices

    Returns:
        A tuple of a list of result for a RBF, lienar and polynomial SVR run
    '''
    dates = np.reshape(dates, (len(dates), 1))

    # see which model best fits our data
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernl='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # train the set
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    # plot the data
    plot_data(dates, prices)

    # return the predicted alerts
    forecast_1, forecast_2, forecast_3 = svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
    return forecast_1, forecast_2, forecast_3


def plot_data(dates, prices):
    '''
    Takes a list of stock dates and prices and plots the data

    Args:
        dates: a list of dates
        prices: a list of prices

    Returns:
        None
    '''
    plt.scatter(dates, prices, color='blue', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='green', label='RBF')
    plt.plot(dates, srv_poly.predict(dates), color='red', label='Polynomial')
    plt.plot(dates, svr_lin.predict(dates), color='black', label='Linear')

    # set the labels
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('SVR')
    plt.legend()

    # show the plot
    plt.show()


def main():
    dates, prices = retrieve_data('aapl.csv')
    forecasts = forecast_prices(dates, price)
    [forecast for forecast in forecasts]


if __name__ == "__main__":
    main()

import numpy as np
from sklearn import datasets, linear_model

from returns import get_data


def run_model():
    # get our data
    X, Y = get_data()

    # grab our linear regeression model
    model = linear_model.LinearRegression()

    # reshape our vectors
    X_vector = X.reshape(-1,1)
    Y_vector = Y.reshape(-1,1)

    # train the model
    model.fit(X_vector, Y_vector)

    # return theta not and theta 1
    return model.intercept_, model.coef_


def main():
    model_result = run_model()
    print(model_result)


if __name__ == '__main__':
    main()

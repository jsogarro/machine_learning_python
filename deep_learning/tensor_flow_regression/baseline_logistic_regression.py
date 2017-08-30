import pandas as pd
import numpy as np
import statsmodels.api as sm

from returns import get_goog_sp500_logistic_data


X, Y = get_goog_sp500_logistic_data()

logit = sm.Logit(Y, X)

result = logit.fit()

predictions = result.predict(X) > 0.5

number_of_accurate_predictions = list(Y == predictions).count(True)

percent_accuracy = float(number_of_accurate_predictions) / float(len(predictions))

print(percent_accuracy)

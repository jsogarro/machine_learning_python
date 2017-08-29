import numpy as np
from sklearn import datasets, linear_model

from returns import get_nasdaq_oil_xom_data

nasdaq, oil, xom = get_nasdaq_oil_xom_data()

# combine our regressors
combined = np.vstack((nasdaq, oil)).T

# create our model
multipleRegressionModel = lienar_model.LinearRegression()

# train teh data on our X and Y values
multipleRegressionModel.fit(combined, xom)
multipleRegressionModel.score(combined, xom)

print(multipleRegressionModel.coef_)
print(multipleRegressionModel.intercept_)

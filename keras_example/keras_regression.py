import tensorflow as tf
import numpy as np
import math

from keras.models import Sequential
from keras.layers.core import Dense, Activation

# create house siseze between 1k and 3.5k
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# compute house price as a function of hte size with some noise added in
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# normalize our values
def normalize(array):
    return (array - array.mean()) / array.std()

# create our training sample
num_train_samples = math.floor(num_house * 0.7)

# training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# construct the NNN
model = Sequential()
model.add(Dense(1, input_shape=(1,), init='uniform', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='sgd')

# train the model
model.fit(train_house_size_norm, train_price_norm, nb_epoch=300)

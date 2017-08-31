import tensorflow as tf
import numpy as np
from returns import get_data


# variables for our weights and bias
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

# set placeholder for our returns
x = tf.placeholder(tf.float32, [None, 1])

# mutiply our vector by the transformation matrix of weights
Wx = tf.matmul(x, W)

# define our regression formula
y_hat = Wx + b

# place holder for our training data
y = tf.placeholder(tf.float32, [None, 1])

# define the cost function
cost = tf.reduce_mean(tf.square(y_hat - y))

#training step
gamma = 1
train_step_ftrl = tf.train.FtrlOptimizer(gamma).minimize(cost)

# get our data
X, Y = get_data()

# number of observations
dataset_size = len(X)


def train_model(number_of_steps, training_step, batch_size=1):

    # initialize
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(number_of_steps):
            if dataset_size == batch_size:
                batch_start_idx = 0
            elif dataset_size < batch_size:
                raise ValueError("data_size %d, must be larger than batch_size %d" % dataset_size, batch_size)
            else:
                batch_start_idx = (i * batch_size) % dataset_size

            batch_end_idx = batch_start_idx + batch_size

            # splice the dataset for our batches
            batch_xs = X[batch_start_idx : batch_end_idx]
            batch_ys = Y[batch_start_idx : batch_end_idx]

            # create our feed dictionary
            feed = { x: batch_xs.reshape(-1,1), y: batch_ys.reshape(-1,1) }

            # run the training step
            sess.run(training_step, feed_dict=feed)

            if (i + 1) % 500 == 0:
                print("W: %f" % sess.run(W))
                print("b: %f" % sess.run(b))
                print("cost: %f" % sess.run(cost, feed_dict=feed))


def main():
    train_model(10000, train_step_ftrl, 100)


if __name__ == '__main__':
    main()

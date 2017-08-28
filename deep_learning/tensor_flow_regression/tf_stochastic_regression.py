import tensorflow as tf
import numpy as np
from returns import get_data


# variables for our weights and bias
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

# set placeholder for our returns
x = tf.placeholder(tf.float32, [None, 1])

# mutiple our vector by the transformation matrix of weights
Wx = tf.matmul(x, W)

# define our regression formula
y_hat = Wx + b

# place holder for our y labels
y = tf.placeholder(tf.float32, [None, 1])

# define the cost function
cost = tf.reduce_mean(tf.square(y_hat - y))

#training step
gamma = 0.1
train_step_constant = tf.train.GradientDescentOptimizer(gamma).minimize(cost)


def train_model(number_of_steps, training_step):
    # get our data
    X, Y = get_data()

    # initialize
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(number_of_steps):
            # get training point
            x_point = [ [ X[i % len(X) ] ] ]
            y_point = [ [ Y[i % len(Y) ] ] ]

            xs = np.array(x_point)
            ys = np.array(y_point)

            # create our feed dictionary
            feed = {x: xs, y_hat: ys}

            # run the training step
            session.run(training_step, feed_dict=feed)


def main():
    train_model(10000, train_step_constant)


if __name__ == '__main__':
    main()

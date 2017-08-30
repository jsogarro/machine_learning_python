import tensorflow as tf
from returns import get_nasdaq_oil_xom_data


# simple linear regression formula y_hat = W1*x1 + W2*x2 + b
nasdaq_W = tf.Variable(tf.zeros([1,1]), name='nasdaq_W')
oil_W = tf.Variable(tf.zeros([1,1]), name='oil_W')

# set a variable for our bias
b = tf.Variable(tf.zeros([1]), name='b')

# set placeholders for our regressors
nasdaq_x = tf.placeholder(tf.float32, [None, 1], name='nasdaq_x')
oil_x = tf.placeholder(tf.float32, [None, 1], name='oil_x')

# matrix multiply our input vector by the transformation matrix of weights
nasdaq_Wx = tf.matmul(nasdaq_x, nasdaq_W)
oil_Wx = tf.matmul(oil_x, oil_W)

# construct our regression formula
y_hat = nasdaq_Wx + oil_Wx + b

# create a placholder for our y values
y_ = tf.placeholder(tf.float32, [None, 1])

# define the cost function
cost = tf.reduce_mean(tf.square(y_hat - y_))

# define the model we want to use for our training step
train_step_ftrl = tf.train.FtrlOptimizer(1).minimize(cost)

# get our data and reshape it for TF
nasdaq, oil, xom = get_nasdaq_oil_xom_data()

x_nasdaq = nasdaq.reshape(-1, 1)
x_oil = oil.reshape(-1, 1)
ys = xom.reshape(-1, 1)

# keep track of the length of our data set
dataset_size = len(oil)


def train_data(number_of_steps, training_step, batch_size):
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(number_of_steps):
            if dataset_size == batch_size:
                batch_start_idx = 0
            elif dataset_size < batch_size:
                raise ValueError("data_size %d, must be larger than batch_size %d" % dataset_size, batch_size)
            else:
                batch_start_idx = (i * batch_size) % dataset_size

            batch_end_idx = batch_start_idx + batch_size

            # calculate our batches by stepping through the data
            batch_x_nasdaq = x_nasdaq[batch_start_idx : batch_end_idx]
            batch_x_oil = x_oil[batch_start_idx : batch_end_idx]
            batch_ys = ys[batch_start_idx : batch_end_idx]

            feed = {nasdaq_x: batch_x_nasdaq, oil_x: batch_x_oil, y_: batch_ys}

            session.run(training_step, feed_dict=feed)

            if (i + 1) % 500 == 0:
                print(i+1)
                print("W1: %f" % session.run(nasdaq_W))
                print("W2: %f" % session.run(oil_W))
                print("b: %f" % session.run(b))
                print("cost: %f" % session.run(cost, feed_dict=feed))


def main():
    train_data(5000, train_step_ftrl, len(oil))


if __name__ == '__main__':
    main()

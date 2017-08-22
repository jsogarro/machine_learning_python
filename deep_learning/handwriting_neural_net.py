import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# pull in our data
mnist = input_data.read_data_sets("MNIST_Data", one_hot=True)

# placeholder for our image data
x = tf.placeholder(tf.float32, shape=[None, 784])

# placeholder for our predicted values
y_ = tf.placeholder(tf.float32, [None, 10])

# specify our weights and balances for training
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# define the cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y))

# use gradient descent to minimize our cost function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize our globals
init = tf.global_variables_initializer()

# create the session
sess = tf.Session()

# init session
sess.run(init)

# perform 1500 training steps
for i in range(1500):
    batch_xs, batch_ys = mnist.train.next_batch(100)

# see how accurate the model is
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

# print the result
print("Accuracy: {0}%".format(test_accuracy * 100.0))


# close the session
sess.close()

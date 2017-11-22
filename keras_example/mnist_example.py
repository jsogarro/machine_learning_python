import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
from keras.metrics import categorical_accuracy as accuracy

# create a TF session
sess = tf.Session()

# pass the session to Keras
K.set_session(sess)

# load our data and enable one-hot encoding
mnist_data = input_data.read_data_sets('MINST_data', one_hot=True)

# create a placeholder for our inputs
img = tf.placeholder(tf.float32, shape=(None, 784))

# add our tensors to Keras
# create a fully-connected layer with 128 units and ReLU for our activation function
x = Dense(128, activation='relu')(img)
x = Dense(128, activation='relu')(x)

# output layer
preds = Dense(10, activation='softmax')(x)

# set up our TF labels
labels = tf.placeholder(tf.float32, shape=(None, 10))

# define the cost function
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# define the training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# initialize our variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# train the model in a tf session using batch gradient descent
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})

# test accuracy of hte model
acc_value = accuracy(labels, preds)
with sess.as_default():
    model_result = acc_value.eval(feed_dict={img: mnist_data.test.images,
                              labels: mnist_data.test.labels})
    print(model_result)

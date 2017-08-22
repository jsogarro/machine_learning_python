import tensorflow as tf


# create a simple constant
hello = tf.constant('Hello, TensorFlow!')

# create a session
sess = tf.Session()

# run the session
def main():
    print(sess.run(hello))

if __name__ == "__main__":
    main()

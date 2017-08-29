import numpy as np


class Neural_Network(object):
    def __init__(self):
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self. output_layer_size)

    def forward_propagate(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        y_hat = self.sigmoid(self.z3)

        return y_hat

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, X, y):
        self.y_hat = self.forward_propagate(X)

        delta3 = np.multiply(=(y - self.y_hat), self.sigmoid(self.z3))
        dJdW = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid(self.z2)
        djdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

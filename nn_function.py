from mpc_function import *


class ThreeLayerNetwork:

    def __init__(self, sigmoid):
        self.sigmoid = sigmoid

    def train(self, X, y, iterations):

        # initial weights
        self.synapse0 = secure(2 * np.random.random((3, 4)) - 1)
        self.synapse1 = secure(2 * np.random.random((4, 1)) - 1)
        # print("X:", X)
        # print("y:", y)
        # print("synapse0:", self.synapse0)

        # training
        for i in range(iterations):

            # forward propagation
            layer0 = X
            layer1 = self.sigmoid.evaluate(np.dot(layer0, self.synapse0))
            layer2 = self.sigmoid.evaluate(np.dot(layer1, self.synapse1))

            # back propagation
            layer2_error = y - layer2
            layer2_delta = layer2_error * self.sigmoid.derive(layer2)
            layer1_error = np.dot(layer2_delta, self.synapse1.T)
            layer1_delta = layer1_error * self.sigmoid.derive(layer1)

            # update
            self.synapse1 += np.dot(layer1.T, layer2_delta)
            self.synapse0 += np.dot(layer0.T, layer1_delta)

    def predict(self, X):
        layer0 = X
        layer1 = self.sigmoid.evaluate(np.dot(layer0, self.synapse0))
        layer2 = self.sigmoid.evaluate(np.dot(layer1, self.synapse1))
        return layer2


class SigmoidMaclaurin5:

    def __init__(self):
        ONE = SecureRational(1)
        W0 = SecureRational(1 / 2)
        W1 = SecureRational(1 / 4)
        W3 = SecureRational(-1 / 48)
        W5 = SecureRational(1 / 480)
        self.sigmoid = np.vectorize(lambda x: W0 + (x * W1) + (x ** 3 * W3) + (x ** 5 * W5))
        self.sigmoid_deriv = np.vectorize(lambda x: (ONE - x) * x)

    def evaluate(self, x):
        return self.sigmoid(x)

    def derive(self, x):
        return self.sigmoid_deriv(x)


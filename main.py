import random
import numpy as np
from mpc_function import *
from nn_function import  *

random.seed(10)
np.random.seed(10)

X_train = np.array([
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]
        ])

y_train = np.array([[
            0,
            0,
            1,
            1
        ]]).T

X_full = np.array([
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]
        ])


def evaluate(network):
    for x in X_full:
        score = reveal(network.predict(secure(x)))
        prediction = 1 if score > 0.5 else 0
        print("Prediction on %s: %s (%.8f)" % (x, prediction, score))


# pick approximation
sigmoid = SigmoidMaclaurin5()

# train
network = ThreeLayerNetwork(sigmoid)
network.train(secure(X_train), secure(y_train), 500)

# evaluate predictions
evaluate(network)






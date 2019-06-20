import numpy as np


def sigmoid(x):
    # Our nonlinear function
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    # Derivative of the sigmoid
    return x * (1 - x)


def forwardPropagate(layer, synapse, nonLinearFn):
    return nonLinearFn(np.dot(layer, synapse))

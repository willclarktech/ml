import numpy as np


def sigmoid(x):
    # Our nonlinear function
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # Derivative of the sigmoid
    return x * (1 - x)


def forward_propagate(layer, synapse, non_linear_fn):
    return non_linear_fn(np.dot(layer, synapse))


def get_layer_width(layer):
    return len(layer[0])


def generate_random_synapse(input_width, output_width):
    return 2 * np.random.random((input_width, output_width)) - 1

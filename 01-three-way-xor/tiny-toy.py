#!/usr/bin/env python
import numpy as np

# Make random numbers deterministic for debugging
np.random.seed(1337)


def sigmoid(x):
    # Our nonlinear function
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    # Derivative of the sigmoid
    return x * (1 - x)


# Input dataset: each row is a training example
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Output dataset: each row is a training example
y = np.array([[0, 0, 1, 1]]).T

# Initialise weights with mean of 0
synapse0 = 2 * np.random.random((3, 1)) - 1

ITERATIONS = 100000

# Training loop
for i in range(ITERATIONS):

    # First (input) layer
    layer0 = X
    # Second (hidden/output) layer
    layer1 = sigmoid(np.dot(layer0, synapse0))

    layer1_error = y - layer1

    if i % (ITERATIONS/10) == 0:
        pass
        # print(layer1_error)

    # Multiply error by slope of sigmoid at values in layer 1
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Update weights in synapse0
    synapse0 += np.dot(layer0.T, layer1_delta)


# Print result
print("Output after training:")
print(layer1)

# E.g.
# [[0.00301777]
#  [0.00246103]
#  [0.99799169]
#  [0.99753711]]

# Time: 1.228s

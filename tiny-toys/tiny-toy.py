#!/usr/bin/env python
import numpy as np
import os

from utils import forwardPropagate, sigmoid, sigmoid_derivative

# Make random numbers deterministic for debugging
np.random.seed(1337)

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

# First (input) layer
layer0 = X

ITERATIONS = int(os.getenv("ITERATIONS", "100000"))
# Training loop
for i in range(ITERATIONS):

    # Second (hidden/output) layer
    layer1 = forwardPropagate(layer0, synapse0, sigmoid)

    layer1_error = y - layer1

    if (os.getenv("DEBUG") != None) & (i % (ITERATIONS/10) == 0):
        print(layer1_error)

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

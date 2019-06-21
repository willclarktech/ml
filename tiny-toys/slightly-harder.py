import numpy as np
import os

from utils import forward_propagate, generate_random_synapse, get_layer_width, sigmoid, sigmoid_derivative

DEBUG = os.getenv("DEBUG") != None

np.random.seed(1337)

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = np.array([[0, 1, 1, 0]]).T

layer0 = X
hidden_width = 4
synapse0 = generate_random_synapse(get_layer_width(layer0), hidden_width)
synapse1 = generate_random_synapse(hidden_width, get_layer_width(y))

ITERATIONS = int(os.getenv("ITERATIONS", "100000"))

# Training loop
for i in range(ITERATIONS):
    layer1 = forward_propagate(layer0, synapse0, sigmoid)
    layer2 = forward_propagate(layer1, synapse1, sigmoid)

    layer2_error = y - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    layer1_error = layer2_delta.dot(synapse1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    synapse1 += layer1.T.dot(layer2_delta)
    synapse0 += layer0.T.dot(layer1_delta)

    if DEBUG & (i % (ITERATIONS/10) == 0):
        print("Error: " + str(np.mean(np.abs(layer2_error))))

# Print result
print("Output after training:")
print(layer2)

# E.g.
# [[0.00162276]
#  [0.99680523]
#  [0.9961384 ]
#  [0.00425631]]

# Time: 2.655s

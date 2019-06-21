import numpy as np
import os

from utils import forward_propagate, generate_random_synapse, get_layer_width, sigmoid, sigmoid_derivative

DEBUG = os.getenv("DEBUG") != None

np.random.seed(1337)

X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])

y = np.array([
    [1],
    [1],
    [1],
    [0],
    [1],
    [0],
    [0],
    [1],
])

layer0 = X
hidden_width = 4
synapse0 = generate_random_synapse(get_layer_width(X), hidden_width)
synapse1 = generate_random_synapse(hidden_width, get_layer_width(y))

iterations = int(os.getenv("ITERATIONS", "100000"))
for i in range(iterations):
    layer1 = forward_propagate(layer0, synapse0, sigmoid)
    layer2 = forward_propagate(layer1, synapse1, sigmoid)

    layer2_error = y - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    layer1_error = np.dot(layer2_delta, synapse1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    synapse1 += np.dot(layer1.T, layer2_delta)
    synapse0 += np.dot(layer0.T, layer1_delta)

    if DEBUG & (i % (iterations/10) == 0):
        print("Error: ", layer2_error)


print("Output after training:")
print(layer2)

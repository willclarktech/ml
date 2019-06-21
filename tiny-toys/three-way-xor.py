import numpy as np
import os
from utils import forwardPropagate, sigmoid, sigmoid_derivative

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
synapse0 = 2 * np.random.random((len(X[0]), hidden_width)) - 1
synapse1 = 2 * np.random.random((hidden_width, len(y[0]))) - 1

iterations = int(os.getenv("ITERATIONS", "100000"))
for i in range(iterations):
    layer1 = forwardPropagate(layer0, synapse0, sigmoid)
    layer2 = forwardPropagate(layer1, synapse1, sigmoid)

    layer2_error = y - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    layer1_error = np.dot(layer2_delta, synapse1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    synapse1 += np.dot(layer1.T, layer2_delta)
    synapse0 += np.dot(layer0.T, layer1_delta)

    if (os.getenv("DEBUG") != None) & (i % (iterations/10) == 0):
        print("Error: ", layer2_error)


print("Output after training:")
print(layer2)

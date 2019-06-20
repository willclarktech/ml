import numpy as np
import os

from utils import forwardPropagate, sigmoid, sigmoid_derivative

np.random.seed(1337)

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = np.array([[0, 1, 1, 0]]).T

input_width = len(X[0])
hidden_width = 4
output_width = len(y[0])
synapse0 = 2 * np.random.random((input_width, hidden_width)) - 1
synapse1 = 2 * np.random.random((hidden_width, output_width)) - 1

ITERATIONS = int(os.getenv("ITERATIONS", "100000"))
layer0 = X

# Training loop
for i in range(ITERATIONS):
    layer1 = forwardPropagate(layer0, synapse0, sigmoid)
    layer2 = forwardPropagate(layer1, synapse1, sigmoid)

    layer2_error = y - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    layer1_error = layer2_delta.dot(synapse1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    synapse1 += layer1.T.dot(layer2_delta)
    synapse0 += layer0.T.dot(layer1_delta)

    if (os.getenv("DEBUG") != None) & (i % (ITERATIONS/10) == 0):
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

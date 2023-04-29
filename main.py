import numpy as np
import random
import math

# parameters
input_size = 2
hidden_size = 2
output_size = 1
init_weight_range = 0.2
beta = 2.0

# input layer: x
# hidden layer: y
# output layer: z
x = np.zeros(input_size)
y = np.zeros(hidden_size)
z = np.zeros(output_size)

x = [1, 0]

# weights
w = np.zeros((input_size, hidden_size))     # input to hidden
v = np.zeros((hidden_size, output_size))    # hidden to output

def init_weights():
    for i in range(hidden_size):
        for j in range(input_size):
            w[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

    for i in range(output_size):
        for j in range(hidden_size):
            v[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

def forward_computation():
    for i in range(hidden_size):
        u = 0
        for j in range(input_size):
            u += w[j, i] * x[i]
        y[i] = 1 / (1 + math.exp(-beta * u))

    for i in range(output_size):
        s = 0
        for j in range(hidden_size):
            s += v[j, i] * y[i]
        z[i] = 1 / (1 + math.exp(-beta * s))

init_weights()
forward_computation()
print(y)
print(z)

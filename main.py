import numpy as np
import random
import math

# parameters
input_size = 2
hidden_size = 2
output_size = 1
init_weight_range = 0.2
beta = 2.0

# data
# (x1, x2) -> y
r = 3.0
data_min_x1 = -3
data_min_x2 = -3
data_max_x1 = 3
data_max_x2 = 3
def data(x1, x2):
    z2 = r * r - x1 * x1 - x2 * x2
    if z2 < 0:
        print("Error: out of sphere")
        return 0
    return math.sqrt(r * r - x1 * x1 - x2 * x2)

# input layer: x
# hidden layer: y
# output layer: z
y = np.zeros(hidden_size)
z = np.zeros(output_size)

# weights
w = np.zeros((input_size, hidden_size))     # input to hidden
v = np.zeros((hidden_size, output_size))    # hidden to output

# sample data
samples_x = []
samples_y = []
def make_sample_data(sample_n):
    for i in range(sample_n):
        sample_x1 = random.uniform(data_min_x1, data_max_x1)
        sample_x2 = random.uniform(data_min_x2, data_max_x2)
        sample_y = data(random.uniform(data_min_x1, data_max_x1), random.uniform(data_min_x2, data_max_x2))
        samples_x.append([sample_x1, sample_x2])
        samples_y.append(sample_y)

def init_weights():
    for i in range(hidden_size):
        for j in range(input_size):
            w[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

    for i in range(output_size):
        for j in range(hidden_size):
            v[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

def forward_computation(x):
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

    return z

def back_propagate():
    for i in range(len(samples_x)):
        output = forward_computation(samples_x[i])
        print(output)

init_weights()
make_sample_data(10)
back_propagate()

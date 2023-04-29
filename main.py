import numpy as np
import random

# parameters
input_size = 2
hidden_size = 2
output_size = 1

init_weight_range = 0.2

# input layer: x
# hidden layer: y
# output layer: z
x = np.zeros(input_size)
y = np.zeros(hidden_size)
z = np.zeros(output_size)

# weights
w = np.zeros((input_size, hidden_size))     # input to hidden
v = np.zeros((hidden_size, output_size))    # hidden to output

def init_weights():
    for i in range(input_size):
        for j in range(hidden_size):
            w[i, j] = random.uniform(-init_weight_range/2, init_weight_range/2)

    for i in range(hidden_size):
        for j in range(output_size):
            v[i, j] = random.uniform(-init_weight_range/2, init_weight_range/2)

# def forward_computation():
#     u = 

init_weights()
print(w)
print(v)

import numpy as np

# parameters
input_size = 2
hidden_size = 2
output_size = 1

# input layer: x
# hidden layer: y
# output layer: z
x = np.zeros(input_size)
y = np.zeros(hidden_size)
z = np.zeros(output_size)

# weights
w = np.zeros((input_size, hidden_size))     # input to hidden
v = np.zeros((hidden_size, output_size))    # hidden to output



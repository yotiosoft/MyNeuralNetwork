import numpy as np
import random
import math
from scipy.stats import mvn
import matplotlib.pyplot as plt

# parameters
input_size = 2
hidden_size = 2
output_size = 1
init_weight_range = 0.2
beta = 2.0
eta = 0.5

# data
# (x1, x2) -> y
r = 3.0
data_min_x1 = -30
data_min_x2 = -30
data_max_x1 = 30
data_max_x2 = 30

mu = np.matrix([0, 0])
sig = np.matrix([[1,0.3],[0.3,1]])

def data(x1, x2):
    datx = np.matrix([x1, x2])
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(datx-mu)*sig.I*(datx-mu).T)
    return [np.exp(b)/a]

# input layer: x
# hidden layer: y
# output layer: z
y = np.zeros(hidden_size)
z = np.zeros(output_size)

# weights
w = np.zeros((hidden_size, input_size))     # input to hidden
v = np.zeros((output_size, hidden_size))    # hidden to output

# sample data
samples_x = []
samples_y = []
def make_sample_data(sample_n):
    for i in range(sample_n):
        sample_x1 = random.uniform(data_min_x1, data_max_x1)
        sample_x2 = random.uniform(data_min_x2, data_max_x2)
        sample_y = data(random.uniform(data_min_x1, data_max_x1), random.uniform(data_min_x2, data_max_x2))[0]
        samples_x.append([sample_x1, sample_x2])
        samples_y.append(sample_y)

def init_weights():
    for i in range(input_size):
        for j in range(hidden_size):
            w[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

    for i in range(hidden_size):
        for j in range(output_size):
            v[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

def forward_computation(x):
    for i in range(input_size):
        u = 0
        for j in range(hidden_size):
            u += w[j, i] * x[i]
        y[j] = 1 / (1 + math.exp(-beta * u))    # u

    for i in range(hidden_size):
        s = 0
        for j in range(output_size):
            s += v[j, i] * y[i]
        z[j] = 1 / (1 + math.exp(-beta * s))    # s

    return z

def back_propagate():
    for n in range(len(samples_x)):
        z = forward_computation(samples_x[n])
        t = data(samples_x[n][0], samples_x[n][1])
        for j in range(hidden_size):
            for k in range(output_size):
                v[k, j] += eta * (t[k] - z[k]) * (z[k] * (1 - z[k])) * y[j]
    
    err_total = 0
    for i in range(len(samples_x)):
        z = forward_computation(samples_x[n])
        t = data(samples_x[n][0], samples_x[n][1])

        for k in range(output_size):
            err_total += t[k] - z[k]

        for i in range(input_size):
            for j in range(hidden_size):
                s = 0
                for k in range(output_size):
                    s += v[k, j] * (t[k] - z[k]) * (z[k] * (1 - z[k]))
                w[j, i] += eta * s * (y[j] * (1 - y[j])) * samples_x[n][i]
    return err_total

init_weights()
make_sample_data(1000)

for i in range(10):
    err_total = back_propagate()
    print("v = " + str(v))
    print("w = " + str(w))
    print("err total: " + str(err_total))

# show figures
plot_sample_x1 = [x[0] for x in samples_x]
plot_sample_x2 = [x[1] for x in samples_x]
plot_sample_y = [y for y in samples_y]

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(plot_sample_x1, plot_sample_x2, plot_sample_y, color='blue')

plt.show()


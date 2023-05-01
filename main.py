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
data_min_x1 = -2
data_min_x2 = -2
data_max_x1 = 2
data_max_x2 = 2

mu = np.matrix([0, 0])
sig = np.matrix([[1,0.01],[0.01,1]])

def data(x1, x2):
    datx = np.matrix([x1, x2])
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(datx-mu)*sig.I*(datx-mu).T)
    return [np.exp(b)/a]

print(data(0, 0))

# weights
w = np.zeros((hidden_size, input_size))     # input to hidden
v = np.zeros((output_size, hidden_size))    # hidden to output

# sample data
def make_sample_data(sample_n):
    ret_x, ret_y = [], []
    for i in range(sample_n):
        sample_x1 = random.uniform(data_min_x1, data_max_x1)
        sample_x2 = random.uniform(data_min_x2, data_max_x2)
        sample_y = data(sample_x1, sample_x2)
        ret_x.append([sample_x1, sample_x2])
        ret_y.append(sample_y)
    return ret_x, ret_y

def init_weights():
    for i in range(input_size):
        for j in range(hidden_size):
            w[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

    for i in range(hidden_size):
        for j in range(output_size):
            v[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

Y = np.zeros(hidden_size)
Z = np.zeros(output_size)
def forward_computation(X):
    for i in range(input_size):
        u = 0
        for j in range(hidden_size):
            u += w[j, i] * X[i]
        Y[j] = 1 / (1 + math.exp(-beta * u))    # u

    for i in range(hidden_size):
        s = 0
        for j in range(output_size):
            s += v[j, i] * Y[i]
        Z[j] = 1 / (1 + math.exp(-beta * s))    # s

    return Z

def back_propagate(samples_x, samples_y):
    for n in range(len(samples_x)):
        z = forward_computation(samples_x[n])
        t = data(samples_x[n][0], samples_x[n][1])
        for j in range(hidden_size):
            for k in range(output_size):
                v[k, j] += eta * (t[k] - Z[k]) * (Z[k] * (1 - Z[k])) * Y[j]
    
    err_total = 0
    for i in range(len(samples_x)):
        z = forward_computation(samples_x[n])
        t = data(samples_x[n][0], samples_x[n][1])

        for k in range(output_size):
            err_total += (t[k] - Z[k])

        for i in range(input_size):
            for j in range(hidden_size):
                s = 0
                for k in range(output_size):
                    s += v[k, j] * (t[k] - Z[k]) * (Z[k] * (1 - Z[k]))
                w[j, i] += eta * s * (Y[j] * (1 - Y[j])) * samples_x[n][i]
    return err_total

init_weights()
samples_x, samples_y = make_sample_data(1000)

# train
for i in range(10):
    err_total = back_propagate(samples_x, samples_y)
    print("v = " + str(v))
    print("w = " + str(w))
    print("err total: " + str(err_total))
print("train done.")

# test
test_data_x, test_data_y = make_sample_data(1000)
test_err_total = 0
for n in range(len(test_data_x)):
    predict = forward_computation(test_data_x[n])
    test_err_total += test_data_y[n] - predict
print("error rate: " + str(test_err_total))

# show figures
plot_sample_x1 = [x[0] for x in samples_x]
plot_sample_x2 = [x[1] for x in samples_x]
plot_sample_y = [y for y in samples_y]

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(plot_sample_x1, plot_sample_x2, plot_sample_y, color='blue')

plt.show()


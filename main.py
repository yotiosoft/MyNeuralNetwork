import numpy as np
import random
import math
from scipy.stats import mvn
import matplotlib.pyplot as plt
import copy

# parameters
input_size = 2 + 1
hidden_size = 4 + 1
output_size = 1
init_weight_range = 0.01
beta = 0.2
eta = 1.0

# data
# (x1, x2) -> y
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

def forward_computation(X):
    y = np.zeros(hidden_size)
    z = np.zeros(output_size)

    for j in range(hidden_size-1):
        u = 0
        for i in range(input_size):
            u += w[j, i] * X[i]
        y[j] = 1 / (1 + math.exp(-beta * u))    # u
    y[hidden_size-1] = 1    # bias

    for k in range(output_size):
        s = 0
        for j in range(hidden_size):
            s += v[k, j] * y[j]
        z[k] = 1 / (1 + math.exp(-beta * s))    # s

    return z, y

def back_propagate(samples_x, samples_y):
    for n in range(len(samples_x)):
        z, y = forward_computation(samples_x[n])
        t = samples_y[n]
        for j in range(hidden_size):
            for k in range(output_size):
                v[k, j] = v[k, j] + eta * (t[k] - z[k]) * (z[k] * (1 - z[k])) * y[j]
    
    err_total = 0
    for n in range(len(samples_x)):
        z, y = forward_computation(samples_x[n])
        t = samples_y[n]

        for k in range(output_size):
            err_total += abs(t[k] - z[k])

        for i in range(input_size):
            for j in range(hidden_size):
                s = 0
                for k in range(output_size):
                    s += v[k, j] * (t[k] - z[k]) * (z[k] * (1 - z[k]))
                w[j, i] = w[j, i] + eta * s * (y[j] * (1 - y[j])) * samples_x[n][i]
    return err_total

init_weights()
samples_x, samples_y = make_sample_data(2000)
train_x = [[1, x[0], x[1]] for x in samples_x[:1000]]
train_y = samples_y[:1000]
test_x = [[1, x[0], x[1]] for x in samples_x[1000:]]
test_y = samples_y[1000:]

# train
for i in range(10000):
    print("Epoch " + str(i))
    err_total = back_propagate(train_x, train_y)
    print("v = " + str(v))
    print("w = " + str(w))
    print("err total: " + str(err_total))
print("train done.")

# test
test_err_total = 0
test_predicted = []
for n in range(len(test_x)):
    predict, _ = forward_computation(test_x[n])
    test_predicted.append(copy.deepcopy(predict))
    test_err_total += abs(test_y[n] - predict)
print("error rate: " + str(test_err_total))

# show figures
plot_train_x1 = [x[1] for x in train_x]
plot_train_x2 = [x[2] for x in train_x]
plot_train_y = [y for y in train_y]

plot_test_x1 = [x[1] for x in test_x]
plot_test_x2 = [x[2] for x in test_x]
plot_test_predicted = [p[0] for p in test_predicted]

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(plot_train_x1, plot_train_x2, plot_train_y, color='blue')

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.scatter(plot_test_x1, plot_test_x2, plot_test_predicted, color='green')

plt.show()

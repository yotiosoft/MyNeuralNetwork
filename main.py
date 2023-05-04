import numpy as np
import random
import math
from scipy.stats import mvn
import matplotlib.pyplot as plt
import copy

class Prameters:
    def __init__(self, input_size, hidden_size, output_size, init_weight_range, beta, eta, train_times, data_func, data_min_x1, data_min_x2, data_max_x1, data_max_x2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weight_range = init_weight_range
        self.beta = beta
        self.eta = eta
        self.train_times = train_times
        self.data_func = data_func
        self.data_min_x1 = data_min_x1
        self.data_min_x2 = data_min_x2
        self.data_max_x1 = data_max_x1
        self.data_max_x2 = data_max_x2

mu = np.matrix([0, 0])
sig = np.matrix([[1,0.01],[0.01,1]])

def gauss(x1, x2):
    datx = np.matrix([x1, x2])
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(datx-mu)*sig.I*(datx-mu).T)
    return [np.exp(b)/a]

def sin4pi(x1, x2):
    return [(1 + np.sin(4*np.pi*x1)) * x2 / 2]

gauss_params = Prameters(2 + 1, 4 + 1, 1, 0.1, 0.2, 1.0, 10000, gauss, -2, -2, 2, 2)
sin4pi_params = Prameters(2 + 1, 5 + 1, 1, 0.1, 0.2, 0.5, 10000, sin4pi, 0, 0, 1, 1)

params = sin4pi_params

# sample data
def make_sample_data(sample_n):
    ret_x, ret_y = [], []
    for i in range(sample_n):
        sample_x1 = random.uniform(params.data_min_x1, params.data_max_x1)
        sample_x2 = random.uniform(params.data_min_x2, params.data_max_x2)
        sample_y = params.data_func(sample_x1, sample_x2)
        ret_x.append([sample_x1, sample_x2])
        ret_y.append(sample_y)
    return ret_x, ret_y

def init_weights():
    for i in range(params.input_size):
        for j in range(params.hidden_size):
            w[j, i] = random.uniform(-params.init_weight_range/2, params.init_weight_range/2)

    for i in range(params.hidden_size):
        for j in range(params.output_size):
            v[j, i] = random.uniform(-params.init_weight_range/2, params.init_weight_range/2)

def forward_computation(x):
    y = np.zeros(params.hidden_size)
    z = np.zeros(params.output_size)

    for j in range(params.hidden_size-1):
        u = 0
        for i in range(params.input_size):
            u += w[j, i] * x[i]
        y[j] = 1 / (1 + math.exp(-params.beta * u))    # u
    y[params.hidden_size-1] = 1    # bias

    for k in range(params.output_size):
        s = 0
        for j in range(params.hidden_size):
            s += v[k, j] * y[j]
        z[k] = 1 / (1 + math.exp(-params.beta * s))    # s

    return z, y

def back_propagate(samples_x, samples_y):
    for n in range(len(samples_x)):
        z, y = forward_computation(samples_x[n])
        t = samples_y[n]
        for j in range(params.hidden_size):
            for k in range(params.output_size):
                v[k, j] = v[k, j] + params.eta * (t[k] - z[k]) * (z[k] * (1 - z[k])) * y[j]
    
    err_total = 0
    for n in range(len(samples_x)):
        z, y = forward_computation(samples_x[n])
        t = samples_y[n]

        for k in range(params.output_size):
            err_total += abs(t[k] - z[k])

        for i in range(params.input_size):
            for j in range(params.hidden_size):
                s = 0
                for k in range(params.output_size):
                    s += v[k, j] * (t[k] - z[k]) * (z[k] * (1 - z[k]))
                w[j, i] = w[j, i] + params.eta * s * (y[j] * (1 - y[j])) * samples_x[n][i]
    return err_total

# weights
w = np.zeros((params.hidden_size, params.input_size))     # input to hidden
v = np.zeros((params.output_size, params.hidden_size))    # hidden to output

init_weights()
samples_x, samples_y = make_sample_data(2000)
train_x = [[1, x[0], x[1]] for x in samples_x[:1000]]
train_y = samples_y[:1000]
test_x = [[1, x[0], x[1]] for x in samples_x[1000:]]
test_y = samples_y[1000:]

# train
for i in range(params.train_times):
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

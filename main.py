import numpy as np
import random
import math
from scipy.stats import mvn
import matplotlib.pyplot as plt
import copy

mu = np.matrix([0, 0])
sig = np.matrix([[1,0.01],[0.01,1]])

class DataPreset:
    def __init__(self, input_size, hidden_size, output_size, init_weight_range, beta, eta, data_func, data_min_x1, data_min_x2, data_max_x1, data_max_x2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weight_range = init_weight_range
        self.beta = beta
        self.eta = eta
        self.data_func = data_func
        self.data_min_x1 = data_min_x1
        self.data_min_x2 = data_min_x2
        self.data_max_x1 = data_max_x1
        self.data_max_x2 = data_max_x2

        # weights
        self.w = np.zeros((self.hidden_size, self.input_size))     # input to hidden
        self.v = np.zeros((self.output_size, self.hidden_size))    # hidden to output

    def data(self, x1, x2):
        return self.data_func(x1, x2)

    def make_sample_data(self, sample_n):
        ret_x, ret_y = [], []
        for _ in range(sample_n):
            sample_x1 = random.uniform(self.data_min_x1, self.data_max_x1)
            sample_x2 = random.uniform(self.data_min_x2, self.data_max_x2)
            sample_y = self.data(sample_x1, sample_x2)
            ret_x.append([sample_x1, sample_x2])
            ret_y.append(sample_y)
        return ret_x, ret_y

    def init_weights(self):
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.w[j, i] = random.uniform(-self.init_weight_range/2, self.init_weight_range/2)

        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.v[j, i] = random.uniform(-self.init_weight_range/2, self.init_weight_range/2)

    def forward_computation(self, x):
        y = np.zeros(self.hidden_size)
        z = np.zeros(self.output_size)

        for j in range(self.hidden_size-1):
            u = 0
            for i in range(self.input_size):
                u += self.w[j, i] * x[i]
            y[j] = 1 / (1 + math.exp(-self.beta * u))    # u
        y[self.hidden_size-1] = 1    # bias

        for k in range(self.output_size):
            s = 0
            for j in range(self.hidden_size):
                s += self.v[k, j] * y[j]
            z[k] = 1 / (1 + math.exp(-self.beta * s))    # s

        return z, y

    def back_propagate(self, samples_x, samples_y):
        for n in range(len(samples_x)):
            z, y = self.forward_computation(samples_x[n])
            t = samples_y[n]
            for j in range(self.hidden_size):
                for k in range(self.output_size):
                    self.v[k, j] = self.v[k, j] + self.eta * (t[k] - z[k]) * (z[k] * (1 - z[k])) * y[j]
        
        err_total = 0
        for n in range(len(samples_x)):
            z, y = self.forward_computation(samples_x[n])
            t = samples_y[n]

            for k in range(self.output_size):
                err_total += abs(t[k] - z[k])

            for i in range(self.input_size):
                for j in range(self.hidden_size):
                    s = 0
                    for k in range(self.output_size):
                        s += self.v[k, j] * (t[k] - z[k]) * (z[k] * (1 - z[k]))
                    self.w[j, i] = self.w[j, i] + self.eta * s * (y[j] * (1 - y[j])) * samples_x[n][i]
        return err_total

def gauss(x1, x2):
    datx = np.matrix([x1, x2])
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(datx-mu)*sig.I*(datx-mu).T)
    return [np.exp(b)/a]

def sin4pi(x1, x2):
    return [(1 + np.sin(4*np.pi*x1)) * x2 / 2]

ds_gauss = DataPreset(2 + 1, 4 + 1, 1, 0.1, 0.2, 1.0, gauss, -2, -2, 2, 2)
ds_sin4pi = DataPreset(2 + 1, 4 + 1, 1, 0.1, 0.2, 1.0, sin4pi, -2, -2, 2, 2)

ds = ds_gauss

ds.init_weights()
samples_x, samples_y = ds.make_sample_data(2000)
train_x = [[1, x[0], x[1]] for x in samples_x[:1000]]
train_y = samples_y[:1000]
test_x = [[1, x[0], x[1]] for x in samples_x[1000:]]
test_y = samples_y[1000:]

# train
for i in range(10000):
    print("Epoch " + str(i))
    err_total = ds.back_propagate(train_x, train_y)
    print("v = " + str(v))
    print("w = " + str(w))
    print("err total: " + str(err_total))
print("train done.")

# test
test_err_total = 0
test_predicted = []
for n in range(len(test_x)):
    predict, _ = ds.forward_computation(test_x[n])
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

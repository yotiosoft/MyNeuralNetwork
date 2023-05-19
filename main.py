import numpy as np
import random
import math
from scipy.stats import mvn
import matplotlib.pyplot as plt
import copy
import sys
import os
import csv

class Prameters:
    def __init__(self, input_size, hidden_size, output_size, init_weight_range, beta, eta, train_times, data_func, data_min_x1, data_min_x2, data_max_x1, data_max_x2, csv_filename):
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
        self.csv_filename = csv_filename

mu = np.matrix([0, 0])
sig = np.matrix([[1,0.01],[0.01,1]])

def gauss(x1, x2):
    datx = np.matrix([x1, x2])
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(datx-mu)*sig.I*(datx-mu).T)
    return [np.exp(b)/a]

def sin4pi(x1, x2):
    return [(1 + np.sin(4*np.pi*x1)) * x2 / 2]

gauss_params = Prameters(2 + 1, 4 + 1, 1, 0.1, 0.2, 1.0, 10000, gauss, -2, -2, 2, 2, "gauss.csv")
sin4pi_params = Prameters(2 + 1, 19 + 1, 1, 0.1, 0.01, 1.0, 10000, sin4pi, 0, 0, 1, 1, "sin4pi.csv")

# hidden 9 beta 0.01 eta 0.8 -> err=25.087

params = sin4pi_params

# args
if len(sys.argv) >= 1:
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == "func":
            if sys.argv[i+1] == "gauss":
                params = gauss_params
            elif sys.argv[i+1] == "sin4pi":
                params = sin4pi_params
        elif sys.argv[i] == "hidden":
            params.hidden_size = int(sys.argv[i+1])
        elif sys.argv[i] == "beta":
            params.beta = float(sys.argv[i+1])
        elif sys.argv[i] == "eta":
            params.eta = float(sys.argv[i+1])
        elif sys.argv[i] == "train":
            params.train_times = int(sys.argv[i+1])

# sample data
def make_sample_data(sample_n):
    ret_x, ret_z = [], []
    for i in range(sample_n):
        sample_x1 = random.uniform(params.data_min_x1, params.data_max_x1)
        sample_x2 = random.uniform(params.data_min_x2, params.data_max_x2)
        sample_z = params.data_func(sample_x1, sample_x2)
        ret_x.append([sample_x1, sample_x2])
        ret_z.append(sample_z)
    return ret_x, ret_z

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

def back_propagate(training_data_x, training_data_y):
    for n in range(len(training_data_x)):
        z, y = forward_computation(training_data_x[n])
        t = training_data_y[n]
        for j in range(params.hidden_size):
            for k in range(params.output_size):
                v[k, j] = v[k, j] + params.eta * (t[k] - z[k]) * (z[k] * (1 - z[k])) * y[j]
    
    err_total = 0
    for n in range(len(training_data_x)):
        z, y = forward_computation(training_data_x[n])
        t = training_data_y[n]

        for k in range(params.output_size):
            err_total += abs(t[k] - z[k])

        for i in range(params.input_size):
            for j in range(params.hidden_size):
                s = 0
                for k in range(params.output_size):
                    s += v[k, j] * (t[k] - z[k]) * (z[k] * (1 - z[k]))
                w[j, i] = w[j, i] + params.eta * s * (y[j] * (1 - y[j])) * training_data_x[n][i]
    return err_total

def train(times, train_X, train_Z):
    err_total_array = []
    for i in range(params.train_times):
        err_total = back_propagate(train_X, train_Z)
        if i % 100 == 0:
            print("Epoch " + str(i))
            print("v = " + str(v))
            print("w = " + str(w))
            print("err total: " + str(err_total))
            err_total_array.append(err_total)
    return err_total_array


def predict(x):
    z, _ = forward_computation(x)
    return z

def output_csv(csv_filename, err_array):
    rows = []
    if os.path.exists(params.csv_filename):
        with open(params.csv_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:
                    rows.append(row)
        rows = list(zip(*rows))
    rows.append(err_array)
    with open(params.csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list(zip(*rows)))

# weights
w = np.zeros((params.hidden_size, params.input_size))     # input to hidden
v = np.zeros((params.output_size, params.hidden_size))    # hidden to output

init_weights()
samples_X, samples_Z = make_sample_data(2000)
train_X = [[1, x[0], x[1]] for x in samples_X[:1000]]
train_Z = samples_Z[:1000]
test_X = [[1, x[0], x[1]] for x in samples_X[1000:]]
test_Z = samples_Z[1000:]

# train
err_array = train(params.train_times, train_X, train_Z)
print("train done.")

# test
test_err_total = 0
test_predicted = []
for n in range(len(test_X)):
    predict_result = predict(test_X[n])
    test_predicted.append(copy.deepcopy(predict_result))
    test_err_total += abs(test_Z[n] - predict_result)
print("error rate: " + str(test_err_total))

# output to csv
output_csv(params.csv_filename, err_array)

# show figures
plot_train_X1 = [x[1] for x in train_X]
plot_train_X2 = [x[2] for x in train_X]
plot_train_Z = [y for y in train_Z]

plot_test_X1 = [x[1] for x in test_X]
plot_test_X2 = [x[2] for x in test_X]
plot_test_predicted = [p[0] for p in test_predicted]

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(plot_train_X1, plot_train_X2, plot_train_Z, color='blue')

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.scatter(plot_test_X1, plot_test_X2, plot_test_predicted, color='green')

plt.show()

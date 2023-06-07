import numpy as np
import random
import math
from scipy.stats import mvn
import matplotlib.pyplot as plt
import copy
import sys
import os
import csv
from sklearn.datasets import load_iris

iris = load_iris()

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

# nonlinear functions
mu = np.matrix([0, 0])
sig = np.matrix([[1,0.01],[0.01,1]])
def gauss(x1, x2):
    datx = np.matrix([x1, x2])
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(datx-mu)*sig.I*(datx-mu).T)
    return [np.exp(b)/a]

def sin4pi(x1, x2):
    return [(1 + np.sin(4*np.pi*x1)) * x2 / 2]

# make sample data
def make_sample_data(data_min_x1, data_min_x2, data_max_x1, data_max_x2, data_func, sample_n):
    ret_x, ret_z = [], []
    for i in range(sample_n):
        sample_x1 = random.uniform(data_min_x1, data_max_x1)
        sample_x2 = random.uniform(data_min_x2, data_max_x2)
        sample_z = data_func(sample_x1, sample_x2)
        ret_x.append([sample_x1, sample_x2])
        ret_z.append(sample_z)
    return ret_x, ret_z

# initialize weights
def init_weights(init_weight_range, input_size, hidden_size, output_size):
    w = np.zeros((hidden_size, input_size))     # input to hidden
    v = np.zeros((output_size, hidden_size))    # hidden to output

    for i in range(input_size):
        for j in range(hidden_size):
            w[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

    for i in range(hidden_size):
        for j in range(output_size):
            v[j, i] = random.uniform(-init_weight_range/2, init_weight_range/2)

    return w, v

# forward computation
# for prediction
def forward_computation(beta, w, v, x):
    input_size = len(x)
    hidden_size = len(w)
    output_size = len(v)

    y = np.zeros(hidden_size)
    z = np.zeros(output_size)

    for j in range(hidden_size-1):
        u = 0
        for i in range(input_size):
            u += w[j, i] * x[i]
        y[j] = 1 / (1 + math.exp(-beta * u))    # u
    y[hidden_size-1] = 1    # bias

    for k in range(output_size):
        s = 0
        for j in range(hidden_size):
            s += v[k, j] * y[j]
        z[k] = 1 / (1 + math.exp(-beta * s))    # s

    return z, y

# back propagation
# for training
def back_propagate(beta, eta, train_x, train_t, w, v):
    input_size = len(train_x[0])
    hidden_size = len(w)
    output_size = len(v)

    for n in range(len(train_x)):
        z, y = forward_computation(beta, w, v, train_x[n])
        t = train_t[n]
        for j in range(hidden_size):
            for k in range(output_size):
                v[k, j] = v[k, j] + eta * (t[k] - z[k]) * (z[k] * (1 - z[k])) * y[j]
    
    err_total = 0
    for n in range(len(train_x)):
        z, y = forward_computation(beta, w, v, train_x[n])
        t = train_t[n]

        for k in range(output_size):
            err_total += abs(t[k] - z[k])

        for i in range(input_size):
            for j in range(hidden_size):
                s = 0
                for k in range(output_size):
                    s += v[k, j] * (t[k] - z[k]) * (z[k] * (1 - z[k]))
                w[j, i] = w[j, i] + eta * s * (y[j] * (1 - y[j])) * train_x[n][i]
    return err_total

# train
# call back_propagate() for train_times
def train(train_times, beta, eta, w, v, train_x, train_z):
    err_array = []
    for i in range(train_times):
        err_total = back_propagate(beta, eta, train_x, train_z, w, v)
        if i % 100 == 0:
            print("Epoch " + str(i))
            print("v = " + str(v))
            print("w = " + str(w))
            print("err total: " + str(err_total))
            err_array.append(err_total)
    return err_array

# predict
# call forward_computation()
def predict(beta, w, v, x):
    z, _ = forward_computation(beta, w, v, x)
    return z

# output to csv file
def output_csv(csv_filename, err_array):
    rows = []
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:
                    rows.append(row)
        rows = list(zip(*rows))
    rows.append(err_array)
    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list(zip(*rows)))

# show figures
# using matplotlib
def show_figures(train_x, train_z, test_x, test_predicted, err_array):
    plot_train_x1 = [x[1] for x in train_x]
    plot_train_x2 = [x[2] for x in train_x]
    plot_train_z = [y for y in train_z]

    plot_test_x1 = [x[1] for x in test_x]
    plot_test_x2 = [x[2] for x in test_x]
    plot_test_predicted = [p[0] for p in test_predicted]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.scatter(plot_train_x1, plot_train_x2, plot_train_z, color='blue')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('z')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    ax2.scatter(plot_test_x1, plot_test_x2, plot_test_predicted, color='green')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('z')

    show_error(err_array)

def show_error(err_array):
    err_left = [i * 100 for i in range(len(err_array))]
    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.plot(err_left, err_array, color='orange')
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("error")

    plt.show()

if __name__ == "__main__":
    # set parameters
    gauss_params = Prameters(2 + 1, 4 + 1, 1, 0.1, 0.2, 0.5, 10000, gauss, -2, -2, 2, 2, "gauss.csv")
    sin4pi_params = Prameters(2 + 1, 19 + 1, 1, 0.1, 0.01, 1.0, 10000, sin4pi, -2, -2, 2, 2, "sincos.csv")
    iris_params = Prameters(4 + 1, 9 + 1, 3, 0.1, 0.5, 0.5, 10000, None, 0, 0, 1, 1, "iris.csv")
    params = iris_params

    # get args
    if len(sys.argv) >= 1:
        for i in range(1, len(sys.argv), 2):
            if sys.argv[i] == "func":
                if sys.argv[i+1] == "gauss":
                    params = gauss_params
                elif sys.argv[i+1] == "sin4pi":
                    params = sin4pi_params
                elif sys.argv[i+1] == "iris":
                    params = iris_params
            elif sys.argv[i] == "hidden":
                params.hidden_size = int(sys.argv[i+1])
            elif sys.argv[i] == "beta":
                params.beta = float(sys.argv[i+1])
            elif sys.argv[i] == "eta":
                params.eta = float(sys.argv[i+1])
            elif sys.argv[i] == "train":
                params.train_times = int(sys.argv[i+1])
            elif sys.argv[i] == "csv":
                params.csv_filename = sys.argv[i+1]

    # initialize weights
    w, v = init_weights(params.init_weight_range, params.input_size, params.hidden_size, params.output_size)

    # make sample data
    if params == gauss_params or params == sin4pi_params:
        samples_x, samples_z = make_sample_data(params.data_min_x1, params.data_min_x2, params.data_max_x1, params.data_max_x2, params.data_func, 2000)
        train_x = [[1, x[0], x[1]] for x in samples_x[:1000]]
        train_z = samples_z[:1000]
        test_x = [[1, x[0], x[1]] for x in samples_x[1000:]]
        test_z = samples_z[1000:]
    elif params == iris_params:
        samples_x, samples_t = iris.data, iris.target
        samples = list(zip(samples_x, samples_t))
        random.shuffle(samples)
        samples_x, samples_t = zip(*samples)
        samples_z = []
        for t in samples_t:
            if t == 0:
                samples_z.append([1, 0, 0])
            elif t == 1:
                samples_z.append([0, 1, 0])
            elif t == 2:
                samples_z.append([0, 0, 1])
        train_x = [[1, x[0], x[1], x[2], x[3]] for x in samples_x[:75]]
        train_z = [[z[0], z[1], z[2]] for z in samples_z[:75]]
        test_x = [[1, x[0], x[1], x[2], x[3]] for x in samples_x[75:]]
        test_z = [[z[0], z[1], z[2]] for z in samples_z[75:]]

    # train
    err_array = train(params.train_times, params.beta, params.eta, w, v, train_x, train_z)
    print("train done.")

    # test
    test_err_total = 0
    test_predicted = []
    l1_tp, l1_fp = 0, 0
    l2_tp, l2_fp = 0, 0
    l3_tp, l3_fp = 0, 0
    for n in range(len(test_x)):
        predict_result = predict(params.beta, w, v, test_x[n])
        test_predicted.append(copy.deepcopy(predict_result))
        test_err_total += abs(test_z[n] - predict_result)
        if params == iris_params:
            answer = np.argmax(test_z[n])
            predicted = np.argmax(predict_result)
            if answer == predicted:
                if answer == 0:
                    l1_tp += 1
                elif answer == 1:
                    l2_tp += 1
                elif answer == 2:
                    l3_tp += 1
            else:
                if answer == 0:
                    l1_fp += 1
                elif answer == 1:
                    l2_fp += 1
                elif answer == 2:
                    l3_fp += 1
    print("error rate: " + str(test_err_total))
    if params == iris_params:
        l1_precision = l1_tp / (l1_tp + l1_fp)
        l2_precision = l2_tp / (l2_tp + l2_fp)
        l3_precision = l3_tp / (l3_tp + l3_fp)
        print("l1 precision: " + str(l1_precision))
        print("l2 precision: " + str(l2_precision))
        print("l3 precision: " + str(l3_precision))
        print("average precision: " + str((l1_precision + l2_precision + l3_precision) / 3))

    # show figures
    if params == gauss_params or params == sin4pi_params:
        show_figures(train_x, train_z, test_x, test_predicted, err_array)
    elif params == iris_params:
        show_error(err_array)
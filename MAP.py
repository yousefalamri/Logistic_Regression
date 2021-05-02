import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

############################# supporting functions for MAP ################################
# Load train.csv and test.csv
with open('bank-note/train.csv') as f:
    training_data = [];
    for line in f:
        terms = line.strip().split(',')
        training_data.append(terms)

with open('bank-note/test.csv') as f:
    testing_data = [];
    for line in f:
        terms = line.strip().split(',')
        testing_data.append(terms)


def convert_to_float(input_data):
    for elem in input_data:
        for i in range(len(input_data[0])):
            elem[i] = float(elem[i])
    return input_data


# augment feature vector
def augment_feature_vector(input_data):
    labels = [elem[-1] for elem in input_data]
    data_list = input_data
    for i in range(len(input_data)):
        data_list[i][-1] = 1.0
    for i in range(len(input_data)):
        data_list[i].append(labels[i])
    return data_list


# convert (0,1) labels to (-1,1)
def convert_to_pm_one(input_data):
    new_list = input_data
    for i in range(len(input_data)):
        if new_list[i][-1] != 1.0:
            new_list[i][-1] = -1.0
    return new_list

training_data = convert_to_float(training_data)  # convert to float  types data
testing_data = convert_to_float(testing_data)

train_data = augment_feature_vector(convert_to_pm_one(training_data))
test_data = augment_feature_vector(convert_to_pm_one(testing_data))

m = len(train_data)
test_len = len(test_data)
ftr_len = len(train_data[0]) - 1

# sign function
def sgn(input):
    sign = 0
    if input > 0:
        sign = 1
    else:
        sign = -1
    return sign

# sigmoid function
def sigmoid(s):
    if s < -100:
        sigma = 0
    else:
        sigma = 1 / (1 + math.e ** (-s))
    return sigma

# counting the number of errors
def error_counter(prediction, actual):
    error_count = 0
    input_length = len(prediction)
    for i in range(input_length):
        if prediction[i] != actual[i]:
            error_count = error_count + 1
    return error_count / input_length


# do prediction and calculate error percentage
def calculate_error(w, input_data):
    prediction_vector = [];
    for i in range(len(input_data)):
        prediction_vector.append(sgn(np.inner(input_data[i][0:len(input_data[0]) - 1], w)))
    actual_labels = [elem[-1] for elem in input_data]
    error_per = error_counter(prediction_vector, actual_labels) * 100.0
    return error_per


def calculate_objective(w, input_data, v):
    logterms = []
    for elem in input_data:
        inexp = -elem[-1] * np.inner(w, elem[0:ftr_len])
        if inexp > 100:
            term1 = inexp
        else:
            term1 = math.log(1 + math.exp(inexp))
        logterms.append(term1)
    term1 = sum(logterms)
    term2 = 1 / (2 * v) * np.inner(w, w)
    return term1 + term2

def calculate_gradient(w, v):
    logterms = []
    for elem in train_data:
        logterms.append(elem[-1] * (1 - sigmoid(elem[-1] * np.inner(w, elem[0:ftr_len]))) * np.asarray(elem[0:ftr_len]))
    return w / v - sum(logterms)


def calculate_stochastic_gradient(w, data_sample, v):
    sig_term = m * data_sample[-1] * (1 - sigmoid(data_sample[-1] * np.inner(w, data_sample[0:ftr_len])))
    return np.asarray([(w[i] / v) - (data_sample[i] * sig_term)  for i in range(ftr_len)])

# rate schedule
def lrn_rate(t, gamma_0, d):
    return gamma_0 / (1 + (gamma_0 / d) * t)


def SGD(w, shfl_vec, v, iter, gamma_0, d):
    w = np.asarray(w)
    L = []
    for i in range(m):
        w = w - lrn_rate(iter, gamma_0, d) * calculate_stochastic_gradient(w, train_data[shfl_vec[i]], v)
        L.append(calculate_objective(w, train_data, v))
        iter = iter + 1
    return [w, L, iter]


def repeat_SGD(w, v, T, gamma_0, d):
    iter = 1
    Loss = []
    for i in range(T):
        shfl_vec = np.random.permutation(m)
        [w, L, iter] = SGD(w, shfl_vec, v, iter, gamma_0, d)
        Loss.extend(L)
        print('epoch # =', i)
    return [w, Loss, iter]




# ############################# Run MAP ################################

def MAP(variances, T):
    d = 2.0
    gamma_0 = 1
    training_error = []
    testing_error = []
    for v in variances:
        w = np.zeros(5)
        [new_w, Loss, count] = repeat_SGD(w, v, T, gamma_0, d)
        training_error.append(calculate_error(new_w, train_data))
        testing_error.append(calculate_error(new_w, test_data))
    return [training_error, testing_error]


vars = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
epochs = 100
[train_error, test_error] = MAP(vars, epochs)
print('training error % =', train_error)
print('testing error % =', test_error)

############################# convergence plot ################################
#
# w0 = np.zeros(ftr_len)
# epochs = 10
# gamma_0 = 2.0
# d = 2.0
# v = 10
# [wt, loss, tt] = repeat_SGD(w0,v, epochs, gamma_0, d)
# plt.plot(loss)
# plt.xlabel('iterations')
# plt.ylabel('Loss')
# plt.title('T= 10')
# plt.show()

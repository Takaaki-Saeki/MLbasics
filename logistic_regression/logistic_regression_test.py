import numpy as np
import csv
from logistic_regression_train import sigmoid_function

def load_weight(file_name):
    weight = []
    f = open(file_name, 'r')
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        weight_tmp = []
        weight_tmp.append(float(row[0]))
        weight_tmp.append(float(row[1]))
        weight.append(weight_tmp)
    return np.mat(weight)

def load_test_data(file_name, n):
    feature_data = []
    f = open(file_name, 'r')
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        feature_tmp = []
        feature_tmp.append(float(row[0]))
        feature_tmp.append(float(row[0]))
        feature_data.append(feature_tmp)
    return np.mat(feature_data)

def predict(data, weight):
    h = sigmoid_function(data * weight.T)
    v = np.shape(h)[0]
    for i in range(v):
        if h[i, 0] < 0.5:
            h[i, 0] = 0.0
        else:
            h[i, 0] = 1.0
    return h


def save_result(file_name, result):
    f = open(file_name, 'w')
    writer = csv.writer(f)
    for i in range(result.shape[0]):
        writer.writerow(str(int(result[i, 0])))


if __name__ == '__main__':
    weight = load_weight('weight.csv')
    n = np.shape(weight)[1]

    test_data = load_test_data('test.csv', n)

    h = predict(test_data, weight)

    save_result('result.csv', h)


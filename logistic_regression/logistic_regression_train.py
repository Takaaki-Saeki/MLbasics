import numpy as np
import csv


def load_training_data(file_name):
    feature_data = []
    target_data = []
    with open('train.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            feature_tmp = []
            feature_tmp.append(float(row[0]))
            feature_tmp.append(float(row[1]))
            feature_data.append(feature_tmp)
            target_data.append(int(row[2]))
        return np.mat(feature_data), np.mat(target_data)

def sigmoid_function(x):
     return 1.0/(1.0 + np.exp(-x))

def bgd_training(feature, target, itr, alpha):
     n = np.shape(feature)[1]
     weight = np.mat(np.ones((n,1)), dtype='float64')
     for i in range(itr):
          h = sigmoid_function(feature * weight)
          err = target - h
          weight = weight + alpha * feature.T * err
     return weight

def save_trained_model(file_name, w):
    with open('weight.csv', 'a') as f:
        writer = csv.writer(f)
        weight_array = []
        for i in range(weight.shape[0]):
            weight_array.append(weight[i,0])
        print(weight_array)
        writer.writerow(weight_array)

if __name__ == '__main__':
     feature, target = load_training_data('train.csv')

     weight = bgd_training(feature, target, 2000, 0.01)

     save_trained_model("weights.csv", weight)


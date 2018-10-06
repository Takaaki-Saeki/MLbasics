import numpy as np
import matplotlib.pyplot as plt


def sigmoid_function(x):
    return 1.0/(1+np.exp(-x))

def grad(data, labels, max_itr=10000, alpha=0.001,):
    data_matrix = np.mat(data)
    label_matrix = np.mat(labels).T

    m,n = np.shape(data_matrix)
    weights = np.ones((n,1))

    for k in range(max_itr):
        h = sigmoid_function(data_matrix*weights)
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.T * error
    return weights

def load_data():
    data = [[1.0,2.0,3.0],[1.0,4.0,3.0],[1.0,2.0,7.0]]
    labels = [0,0,1]
    return data, labels


def out_result():
    data, labels = load_data()
    weights = grad(data, labels)
    print(weights)
    best_fit_plot(weights)


def best_fit_plot(weights):
    data, labels = load_data()
    data_array = np.array(data)
    n = np.shape(data_array)[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(n):
        if int(labels[i]) == 1:
            x1.append(data_array[i,1])
            y1.append(data_array[i,2])
        else:
            x2.append(data_array[i,1])
            y2.append(data_array[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1,c='red')
    ax.scatter(x2, y2,c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-float(weights[0])-float(weights[1])*x)/float(weights[2])
    ax.plot(x,y)
    plt.show()

if __name__ == '__main__':
    out_result()







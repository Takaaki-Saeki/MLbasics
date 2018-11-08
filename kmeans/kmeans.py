import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data():

    path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'
    data = 'iris.data'

    df = pd.read_table(path + data, sep=',', header=None)
    df.drop([4], axis=1, inplace=True)

    return df

def export_to_array(df1):

    arr1 = df1.values
    return arr1

def kmeans(array,k, n):
    cent_index = np.random.choice(array.shape[0], k)
    centroids = array[cent_index]

    new_centroids = np.zeros((k, n))

    while(True):
        class_array = np.zeros(array.shape[0])
        for i in range(array.shape[0]):
            distance_list = []
            for cent in centroids:
                distance = np.linalg.norm(array[i] - cent)
                distance_list.append(distance)
            min_class = distance_list.index(min(distance_list))
            class_array[i] = min_class

        for j in range(k):
            class_idx = np.where(class_array == j)
            new_centroids[j] = array[class_idx].mean(axis=0)

        if np.sum(centroids == new_centroids):
            print('calculation finished!')
            break

        centroids = new_centroids

    return class_array



if __name__ == '__main__':

    df = load_data()

    array = export_to_array(df)

    class_array = kmeans(array, k, n)

    x = np.arange(array.shape[0])
    y = class_array

    plt.plot(x, y)
    plt.show()






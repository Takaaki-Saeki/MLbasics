import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter

def load_data():

    path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'
    data = 'iris.data'

    df = pd.read_table(path + data, sep=',', header=None)
    df.loc[(df[4] == 'Iris-setosa'), 4] = 0
    df.loc[(df[4] == 'Iris-versicolor'), 4] = 1
    df.loc[(df[4] == 'Iris-virginica'), 4] = 2

    return df


def train_test_split(df, num):
    dev_num = round(df.shape[0]/num)
    test_index = np.random.choice(df.shape[0], dev_num)
    train_index = np.array([x for x in df.index if x not in test_index])
    train_df = df.loc[train_index, :]
    test_df = df.loc[test_index, :]
    test_ans = np.array(test_df[4])
    test_df.drop([4], axis=1, inplace=True)

    return train_df, test_df, test_ans


def export_to_array(df1, df2):
    arr1 = df1.values
    arr2 = df2.values

    return arr1, arr2


def knn(train_arr, test_arr, k):

    class_list = []

    for test in test_arr:
        train_list = []
        s = [0, 0, 0]
        for train in train_arr:
            distance = np.linalg.norm(test - train[0:4])
            data_distance_list = [distance, train[4]]
            train_list.append(data_distance_list)
            train_list.sort(key=itemgetter(0))

        for i in range(k):
            for j in range(3):
                if train_list[i][1] == j:
                    s[j] = s[j] + 1

        class_list.append(np.argmax(s))

    return class_list


def validation(class_list, test_ans):

    correct_num = 0
    class_array = np.array(class_list)

    for i in range(class_array.shape[0]):
        if class_array[i] == test_ans[i]:
            correct_num += 1

    correct_rate = correct_num / class_array.shape[0]

    return correct_rate


if __name__ == '__main__':

    df = load_data()

    precision_list = []

    for k in range(1, 31):
        rate_list = []
        for itr in range(3):
            train_df, test_df, test_ans = train_test_split(df, 3)
            train_arr, test_arr = export_to_array(train_df, test_df)
            class_list = knn(train_arr, test_arr, k)
            corr_rate = validation(class_list, test_ans)
            rate_list.append(corr_rate)
        precision_list.append(sum(rate_list)/len(rate_list))

    x = np.arange(1, 31, 1)
    y = np.array(precision_list)

    plt.plot(x, y)
    plt.title('k-precision graph')
    plt.show()


























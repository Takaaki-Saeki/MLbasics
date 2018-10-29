import numpy as np
import pandas as pd
import pickle

class Node:
    def __init__(self, num=None, feature=None, value=None, right=None, left=None, ls=None):
        self.num = num
        self.feature = feature
        self.value = value
        self.right = right
        self.left = left
        self.ls = ls

class TrainData:
    def __init__(self, x, y, cl):
        self.x = x
        self.y = y
        self.cl = cl

class TestData:
    def __init__(self, x, cl):
        self.x = x
        self.cl = cl

def split_tree(data, feature, value):
    # 木を分割する関数

    node_g = []
    node_l = []

    for d in data:
        if d.x[feature] > value:
            node_g.append(d)
        else:
            node_l.append(d)

    return node_g, node_l

def test_data_generation(path=None, feature_num=5, class_num=2):
    # テストデータを生成する関数
    data = []
    test = pd.read_csv(path)
    for i in range(test.shape[0]):
        x = test.loc[i].values[0:feature_num-1]
        d = TestData(x, class_num)
        data.append(d)

    return data


def distribute(d, node_list, leaf, n):
    # 各データを学習モデルによって葉に分配する関数

    print(n)

    for i in range(len(leaf)):
        if n == leaf[0][i].num:
            return leaf[1][i].y

    node_count = 0
    for node in node_list:
        if node.num == n:
            node = node
        else:
            node_count += 1

    if node_count == len(node_list):
        return []

    if d.x[node.feature] > node.value:
        return distribute(d, node_list, leaf, 2*n)
    else:
        return distribute(d, node_list, leaf, 2 * n + 1)


if __name__ == '__main__':
    data = test_data_generation('test.csv')

    f = open('node.bin', 'rb')
    node_list = pickle.load(f)
    f.close()

    g = open('leaf.bin', 'rb')
    leaf = pickle.load(g)
    g.close()

    print(node_list[0])

    class_list = []
    for d in data:
        class_list.append(distribute(d, node_list, leaf, 1))

    y_pred = class_list
    print(class_list)

    y_pred = pd.DataFrame({np.array(y_pred)})
    y_pred.to_csv('result.csv', index=True)

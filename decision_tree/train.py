# 他クラス分類用のdecision treeを学習するprogram
# dataはlistであり、各要素がclass Dataに属するような形式で与えること。
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


def calc_gini(data, feature, value):
    # gini係数を計算する関数

    tmp1 = []
    tmp2 = []
    for d in data:
        if d.x[feature] > value:
            tmp1.append(d)
        else:
            tmp2.append(d)

    if(len(tmp1) > 0) & (len(tmp2) > 0):
        num_tmp1 = []
        num_tmp2 = []
        for i in range(data[0].cl):
            num_tmp1.append(len([d for d in tmp1 if d.y == i]))
            num_tmp2.append(len([d for d in tmp2 if d.y == i]))

        p_tmp1 = list(map(lambda x: (x / len(tmp1))**2, num_tmp1))
        p_tmp2 = list(map(lambda x: (x / len(tmp2))**2, num_tmp2))

        gini = (1 - sum(p_tmp1))*len(tmp1)/len(data) + (1 - sum(p_tmp2))*len(tmp2)/len(data)

    elif len(tmp1) == 0:
        num_tmp2 = []
        for i in range(data[0].cl):
            num_tmp2.append(len([d for d in tmp2 if d.y == i]))

        p_tmp2 = list(map(lambda x: (x / len(tmp2))**2, num_tmp2))

        gini = (1 - sum(p_tmp2))*len(tmp2)/len(data)

    elif len(tmp2) == 0:
        num_tmp1 = []
        for i in range(data[0].cl):
            num_tmp1.append(len([d for d in tmp1 if d.y == i]))

        p_tmp1 = list(map(lambda x: (x / len(tmp1))**2, num_tmp1))

        gini = (1 - sum(p_tmp1))*len(tmp1)/len(data)

    return gini


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


def decision_feature(data):
    # gini係数が最小となるようなfeature, valueの値を決定する関数

    # gini係数、feature、valueの初期値を与えておく
    min_gini = float('inf')
    min_feature = 0
    min_val = data[0].x[min_feature]

    # gini係数が最小となるようなfeature、valueの値を計算する
    # 全データの値についてループを回す
    for d in data:
        for feature in range(len(d.x)):
            value = d.x[feature]
            gini = calc_gini(data, feature, value)
            if gini < min_gini:
                min_gini = gini
                min_feature = feature
                min_val = value

    return min_feature, min_val


def build_tree(data, n, node_list):
    # decision treeを生成する関数

    feature, value = decision_feature(data)
    node = Node(n, feature, value, 2 * n, 2 * n + 1, data)
    node_list.append(node)

    if (len(data) == 0) or(len(data) == 1):
        return node_list

    init_y = data[0].y
    if len([d for d in data if d.y == init_y]) == len(data):
        return node_list

    node_g, node_l = split_tree(data, feature, value)
    print(len(node_g), len(node_l))
    print('feature:{}, value:{}'.format(feature, value))
    for d in node_g:
        print('right node x:{}, y:{}'.format(d.x, d.y))
    for d in node_l:
        print('left node x:{}, y:{}'.format(d.x, d.y))

    build_tree(node_g, node.right, node_list)
    build_tree(node_l, node.left, node_list)


def train_data_generation(path=None,feature_num=5, class_num=2):
    # csvファイルから関数に入れられる形のdataに変換する関数

    data = []
    train = pd.read_csv(path)
    for i in range(train.shape[0]):
        x = train.loc[i].values[0:feature_num-1]
        y = train.loc[i].values[feature_num]
        d = TrainData(x, y, class_num)
        data.append(d)

    return data


def search_leaf(n, node_list, leaf_ls):
    # 決定木の葉となるノードのノード番号リストを返す関数
    num_ls = []
    for node in node_list:
        num_ls.append(node.num)

    if (2*n in num_ls) or (2*n+1 in num_ls):
        search_leaf(2*n, node_list, leaf_ls)
        search_leaf(2*n+1, node_list, leaf_ls)
    else:
        leaf_ls.append(n)

    leaf = []
    for n in node_list:
        for l in leaf_ls:
            if n.num == l:
                leaf.append(n)

    return leaf


def specify_class(leaf_ls):
    # 各leafとそのclassを対応させる関数
    leaf_cl = []
    for leaf in leaf_ls:
        leaf_cl.append(leaf.ls[0])

    leaf_all = []
    leaf_all.append(leaf_ls)
    leaf_all.append(leaf_cl)

    return leaf_all


if __name__ == '__main__':

    data = train_data_generation('train.csv')
    node_list = []
    build_tree(data, 1, node_list)

    leaf_ls = []
    leaf_ls = search_leaf(1, node_list, leaf_ls)
    leaf = specify_class(leaf_ls)

    f = open('node.bin', 'wb')
    pickle.dump(node_list, f)
    f.close()

    g = open('leaf.bin', 'wb')
    pickle.dump(leaf, g)
    g.close()

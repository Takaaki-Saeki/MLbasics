import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(val1, val2):
    # auto-mpgデータを読み込む関数
    data = pd.read_csv('auto-mpg.csv')
    df = data[[val1, val2, 'mpg']]

    return df


def compensate(df):
    # horsepower列に含まれる欠損地を中央値で補完する関数

    horse_median = df.loc[df['horsepower']!='?', 'horsepower'].median()
    df.loc[df['horsepower']=='?', :] = horse_median
    df['horsepower'] = df['horsepower'].astype(float)

    return df


def scaling(df, val1, val2):
    # 説明変数を正規化する関数
    df[val1] = (df[val1] - df[val1].mean()) / df[val1].std()
    df[val2] = (df[val2] - df[val2].mean()) / df[val2].std()

    return df


def linear_regression(df, data_num, val1, val2):
    # 線形重回帰を行う関数

    v1 = np.array(df[val1])
    v2 = np.array(df[val2])
    m = np.array(df['mpg'])
    ones = np.ones(data_num)

    X = np.array([v1, v2, ones]).T

    W = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, m))

    x, y = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-2, 2, 60))
    z1 = W[0]*x + W[1]*y + W[2]

    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig)
    ax.scatter(v1, v2, m)
    ax.plot_wireframe(x, y, z1, color='red')
    ax.set_xlabel('{} (scaled)'.format(val1))
    ax.set_ylabel('{} (scaled)'.format(val2))
    ax.set_zlabel('mpg')
    plt.savefig('result1_{}_{}.jpg'.format(val1, val2))
    plt.show()


def dimension2_regression(df, data_num, val1, val2):
    # 説明変数の多項式化により2次の重回帰を行う関数
    v1 = np.array(df[val1])
    v2 = np.array(df[val2])
    m = np.array(df['mpg'])
    ones = np.ones(data_num)

    X = np.array([v1, v2, v1*v2, v1**2, v2**2, ones]).T

    W = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, m))

    x, y = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-2, 2, 60))
    z1 = W[0]*x + W[1]*y + W[2]*x*y +W[3]*x**2 + W[4]*y**2 + W[5]

    fig = plt.figure(figsize=(7,6))
    ax = Axes3D(fig)
    ax.scatter(v1, v2, m.T)
    ax.plot_wireframe(x, y, z1, color='red')
    ax.set_xlabel('{} (scaled)'.format(val1))
    ax.set_ylabel('{} (scaled)'.format(val2))
    ax.set_zlabel('mpg')
    plt.savefig('result2_{}_{}.jpg'.format(val1, val2))
    plt.show()


if __name__ == '__main__':

    val1, val2 = ('horsepower', 'weight')
    df = load_data(val1, val2)
    if (val1 == 'horsepower') or (val2 == 'horsepower'):
        df = compensate(df)

    df = scaling(df, val1, val2)
    dimension2_regression(df, 398, val1, val2)









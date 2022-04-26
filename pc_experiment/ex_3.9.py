import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from skspatial.objects import Line
from skspatial.objects import Vector
from scipy.stats import norm
from ml_utils import plot_decision_boundaries


def cal_cov_and_avg(samples):
    """
    给定一个类别的数据，计算协方差矩阵和平均向量
    :param samples: 样本
    :return: 方差和均值
    """

    # 计算均值
    mu = np.mean(samples, axis=0)
    # 计算协方差，根据公式97得
    cov = np.cov(samples - mu, bias=True, rowvar=False) * samples.shape[0]

    return cov, mu


def fisher(c_1, c_2):
    """
    两类情况的Fisher算法
    :param c_1: 类别1的样本
    :param c_2: 类别2的样本
    :return: 返回方向w
    """

    # 第一步：计算均值和方差
    cov_1, mu_1 = cal_cov_and_avg(c_1)
    cov_2, mu_2 = cal_cov_and_avg(c_2)
    # 第二步：计算总类内散度矩阵S_w，根据公式98得
    s_w = cov_1 + cov_2
    # 计算总类间散度矩阵，根据公式102得
    s_b = np.dot(np.transpose(mu_1 - mu_2), mu_1 - mu_2)

    # 第三步：计算S_w的逆矩阵
    # 第四步：计算w，根据公式106得
    w = np.dot(np.linalg.inv(s_w), mu_1 - mu_2)

    return w


def cal_angle_of_vector(v0, v1, is_use_deg=True):
    dot_product = np.dot(v0, v1)
    v0_len = np.linalg.norm(v0)
    v1_len = np.linalg.norm(v1)
    try:
        angle_rad = np.arccos(dot_product / (v0_len * v1_len))
    except ZeroDivisionError as error:
        raise ZeroDivisionError("{}".format(error))

    if is_use_deg:
        return np.rad2deg(angle_rad)
    return angle_rad


# 求两个点形成的向量和投影向量的夹角
def cal_angle_of_point(converted_point, point, w):
    distance = [converted_point[0] - point[0],
                converted_point[1] - point[1],
                converted_point[2] - point[2]]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2 + distance[2] ** 2)
    direction = [distance[0] / norm, distance[1] / norm, distance[2] / norm]

    return cal_angle_of_vector(w, direction)


def program_2(data):
    w2_data = data.loc[data['y'] == 2, 'x1':'x3'].to_numpy()
    w3_data = data.loc[data['y'] == 3, 'x1':'x3'].to_numpy()

    w = fisher(w2_data, w3_data)

    print('P129 Prb9 b题结果')
    print('W2和W3在使用Fisher进行投影的方向 w={}'.format(w))


def program_3(data):
    w2_data = data.loc[data['y'] == 2, 'x1':'x3'].to_numpy()
    w3_data = data.loc[data['y'] == 3, 'x1':'x3'].to_numpy()

    # 计算W
    w = fisher(w2_data, w3_data)

    line = Line([0, 0, 0], w)

    w2_converted_dataset = None
    w3_converted_dataset = None
    for i, d in enumerate(zip(w2_data, w3_data)):
        w2_point = Vector(d[0])
        w3_point = Vector(d[1])

        # 得到w2_point在w上的投影
        w2_converted = line.project_vector(w2_point)
        w3_converted = line.project_vector(w3_point)

        if w2_converted_dataset is None or w3_converted_dataset is None:
            w2_converted_dataset = w2_converted
            w3_converted_dataset = w3_converted

            # 验证投影直线与w夹角为90度
            deg_w2 = cal_angle_of_point(w2_converted_dataset, w2_point, w)
            deg_w3 = cal_angle_of_point(w3_converted_dataset, w3_point, w)
            print('deg_{}_w2 = {:.2f}'.format(i, deg_w2))
            print('deg_{}_w3 = {:.2f}'.format(i, deg_w3))
        else:
            # 验证投影直线与w夹角为90度
            deg_w2 = cal_angle_of_point(w2_converted, w2_point, w)
            deg_w3 = cal_angle_of_point(w3_converted, w3_point, w)
            print('deg_{}_w2 = {:.2f}'.format(i, deg_w2))
            print('deg_{}_w3 = {:.2f}'.format(i, deg_w3))

            w2_converted_dataset = np.vstack((w2_converted_dataset,
                                              line.project_vector(w2_point)))
            w3_converted_dataset = np.vstack((w3_converted_dataset,
                                              line.project_vector(w3_point)))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    w2_dataset = data.loc[data['y'] == 2, 'x1':'x3'].to_numpy()
    w3_dataset = data.loc[data['y'] == 3, 'x1':'x3'].to_numpy()

    # 绘制原始数据点
    ax.scatter(w2_dataset[:, 0], w2_dataset[:, 1], w2_dataset[:, 2],
               marker='o', label='类别-2', color='green')
    ax.scatter(w3_dataset[:, 0], w3_dataset[:, 1], w3_dataset[:, 2],
               marker='^', label='类别-3', color='red')

    # 绘制投影后的点
    ax.scatter(w2_converted_dataset[:, 0],
               w2_converted_dataset[:, 1],
               w2_converted_dataset[:, 2],
               marker='o', label='投影后的类别-2',
               color='green')
    ax.scatter(w3_converted_dataset[:, 0],
               w3_converted_dataset[:, 1],
               w3_converted_dataset[:, 2],
               marker='^', label='投影后的类别-3',
               color='red')

    # 绘制最优w的直线
    plt.plot([w[0] * -2, w[0] * 2],
             [w[1] * -2, w[1] * 2],
             [w[2] * -2, w[2] * 2],
             label='最优方向$w$的直线')

    # 绘制投影线
    for points in zip(w2_dataset, w2_converted_dataset):
        plt.plot([points[0][0], points[1][0]],
                 [points[0][1], points[1][1]],
                 [points[0][2], points[1][2]], color='green',
                 linestyle="--")
    for points in zip(w3_dataset, w3_converted_dataset):
        plt.plot([points[0][0], points[1][0]],
                 [points[0][1], points[1][1]],
                 [points[0][2], points[1][2]], color='red',
                 linestyle="--")

    ax.set_xlabel('X1 Label')
    ax.set_ylabel('X2 Label')
    ax.set_zlabel('X3 Label')

    plt.title('最优方向的W')
    plt.legend()
    plt.show()


def program_4(data):
    w2_data = data.loc[data['y'] == 2, 'x1':'x3'].to_numpy()
    w3_data = data.loc[data['y'] == 3, 'x1':'x3'].to_numpy()

    # 计算W
    w = fisher(w2_data, w3_data).reshape((1, 3))

    # 计算投影后的数据点
    w2_projection_data = []
    w3_projection_data = []
    for i, point in enumerate(zip(w2_data, w3_data)):
        w2_point = point[0].reshape((3, 1))
        w3_point = point[1].reshape((3, 1))

        w2_converted = np.dot(w, w2_point)
        w3_converted = np.dot(w, w3_point)

        w2_projection_data = np.append(w2_projection_data, w2_converted)
        w3_projection_data = np.append(w3_projection_data, w3_converted)

    # 构建数据集
    X = np.append(w2_projection_data, w3_projection_data).reshape(-1, 1)
    y = np.append(np.full([1, w2_projection_data.shape[0]], 2), np.full([1, w3_projection_data.shape[0]], 3))

    # 创建高斯密度估计的贝叶斯模型
    gnb = GaussianNB()
    # 拟合数据
    gnb.fit(X, y)
    # 计算每类的方差
    sigma = gnb.var_
    # 计算每类的均值
    mu = gnb.theta_

    # 决策点，
    boundary = (abs(mu[0] - mu[1]) / 2 + np.min(mu))[0]

    print('P129 Prb9 d题结果')
    print('决策点为 = {:0.2f}'.format(boundary))

    # 计算每个分类的概率分布
    w2_x = np.linspace(-1, 1, 100)
    w3_x = np.linspace(-1, 1, 100)
    w2_y = norm.pdf(w2_x, mu[0], sigma[0])
    w2_mu_y = norm.pdf(mu[0], mu[0], sigma[0])
    w3_y = norm.pdf(w2_x, mu[1], sigma[1])
    w3_mu_y = norm.pdf(mu[1], mu[1], sigma[1])
    # 绘制w2的正态曲线
    plt.plot(w2_x, w2_y, label='$w_{2}$投影后的高斯分布', color='green')
    plt.plot(w3_x, w3_y, label='$w_{3}$投影后的高斯分布', color='red')
    plt.annotate('$w_{2}$投影后的高斯分布',
                 [mu[0], w2_mu_y - 2],
                 xytext=(mu[0] + 0.4, w2_mu_y + 1),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
                 fontsize=14,
                 horizontalalignment='right',
                 verticalalignment='top')
    plt.annotate('$w_{3}$投影后的高斯分布',
                 [mu[1], w3_mu_y - 2],
                 xytext=(mu[1] - 0.2, w2_mu_y + 1),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
                 fontsize=14,
                 horizontalalignment='right',
                 verticalalignment='top')

    # 绘制决策边界
    x_min, x_max = X.min() - 0.1, X.max() + 0.1
    y_min, y_max = -1, 30
    np.linspace(x_min, x_max, 100)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = gnb.predict(xx.ravel().reshape(-1, 1))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    # 绘制决策
    plt.plot([boundary, boundary], [0, 40], label='决策边界点={:.2f}'.format(boundary))
    plt.annotate('决策边界点={:.2f}'.format(boundary),
                 [boundary, 40],
                 xytext=(boundary + 0.3, 40 + 0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
                 fontsize=14,
                 horizontalalignment='right',
                 verticalalignment='top')

    # 画投影点
    plt.scatter(w2_projection_data, np.zeros(w2_projection_data.shape[0]),
                marker='o', label='投影后的类别-2', color='green')
    plt.scatter(w3_projection_data, np.zeros(w3_projection_data.shape[0]),
                marker='^', label='投影后的类别-3', color='red')

    # 绘制一条直线
    plt.plot([-0.5, 0.5], [0, 0])

    plt.title('分类决策面')
    plt.legend()
    plt.show()


def program_5(data):
    w2_data = data.loc[data['y'] == 2, 'x1':'x3'].to_numpy()
    w3_data = data.loc[data['y'] == 3, 'x1':'x3'].to_numpy()

    # 计算W
    w = fisher(w2_data, w3_data).reshape((1, 3))

    # 计算投影后的数据点
    w2_projection_data = []
    w3_projection_data = []
    for i, point in enumerate(zip(w2_data, w3_data)):
        w2_point = point[0].reshape((3, 1))
        w3_point = point[1].reshape((3, 1))

        w2_converted = np.dot(w, w2_point)
        w3_converted = np.dot(w, w3_point)

        w2_projection_data = np.append(w2_projection_data, w2_converted)
        w3_projection_data = np.append(w3_projection_data, w3_converted)

    # 构建数据集
    X = np.append(w2_projection_data, w3_projection_data).reshape(-1, 1)
    y = np.append(np.full([1, w2_projection_data.shape[0]], 2), np.full([1, w3_projection_data.shape[0]], 3))

    # 创建高斯密度估计的贝叶斯模型
    gnb = GaussianNB()
    # 拟合数据
    gnb.fit(X, y)
    # 计算每类的方差
    sigma = gnb.var_
    # 计算每类的均值
    mu = gnb.theta_

    score = gnb.score(X, y)

    print('P129 Prb9 e题结果')
    print('分类正确率为 = {:0.2f}，误差率为 = {:0.2f} '.format(score, 1 - score))


def program_6(data):
    w2_data = data.loc[data['y'] == 2, 'x1':'x3'].to_numpy()
    w3_data = data.loc[data['y'] == 3, 'x1':'x3'].to_numpy()

    # 非最优方向的W
    w = [1.0, 2.0, -1.5]
    line = Line([0, 0, 0], w)

    w2_converted_dataset = None
    w3_converted_dataset = None
    for i, d in enumerate(zip(w2_data, w3_data)):
        w2_point = Vector(d[0])
        w3_point = Vector(d[1])

        # 得到w2_point在w上的投影
        w2_converted = line.project_vector(w2_point)
        w3_converted = line.project_vector(w3_point)

        if w2_converted_dataset is None or w3_converted_dataset is None:
            w2_converted_dataset = w2_converted
            w3_converted_dataset = w3_converted

            # 验证投影直线与w夹角为90度
            deg_w2 = cal_angle_of_point(w2_converted_dataset, w2_point, w)
            deg_w3 = cal_angle_of_point(w3_converted_dataset, w3_point, w)
            print('deg_{}_w2 = {:.2f}'.format(i, deg_w2))
            print('deg_{}_w3 = {:.2f}'.format(i, deg_w3))
        else:
            # 验证投影直线与w夹角为90度
            deg_w2 = cal_angle_of_point(w2_converted, w2_point, w)
            deg_w3 = cal_angle_of_point(w3_converted, w3_point, w)
            print('deg_{}_w2 = {:.2f}'.format(i, deg_w2))
            print('deg_{}_w3 = {:.2f}'.format(i, deg_w3))

            w2_converted_dataset = np.vstack((w2_converted_dataset,
                                              line.project_vector(w2_point)))
            w3_converted_dataset = np.vstack((w3_converted_dataset,
                                              line.project_vector(w3_point)))

    # 计算投影后的数据点
    w2_projection_data = []
    w3_projection_data = []
    for i, point in enumerate(zip(w2_data, w3_data)):
        w2_point = point[0].reshape((3, 1))
        w3_point = point[1].reshape((3, 1))

        w2_converted = np.dot(w, w2_point)
        w3_converted = np.dot(w, w3_point)

        w2_projection_data = np.append(w2_projection_data, w2_converted)
        w3_projection_data = np.append(w3_projection_data, w3_converted)

    # 构建数据集
    X = np.append(w2_projection_data, w3_projection_data).reshape(-1, 1)
    y = np.append(np.full([1, w2_projection_data.shape[0]], 2), np.full([1, w3_projection_data.shape[0]], 3))

    # 创建高斯密度估计的贝叶斯模型
    gnb = GaussianNB()
    # 拟合数据
    gnb.fit(X, y)
    # 计算每类的方差
    sigma = gnb.var_
    # 计算每类的均值
    mu = gnb.theta_

    # 模型正确率
    score = gnb.score(X, y)

    print('P129 Prb9 f题结果')
    print('分类正确率为 = {:0.2f}，误差率为 = {:0.2f} '.format(score, 1 - score))

    # 2D投影决策面
    # 绘制决策边界
    x_min, x_max = X.min() - 0.1, X.max() + 0.1
    y_min, y_max = -1, 30
    np.linspace(x_min, x_max, 100)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = gnb.predict(xx.ravel().reshape(-1, 1))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)

    # 画投影点
    plt.scatter(w2_projection_data, np.zeros(w2_projection_data.shape[0]),
                marker='o', label='投影后的类别-2', color='green')
    plt.scatter(w3_projection_data, np.zeros(w3_projection_data.shape[0]),
                marker='^', label='投影后的类别-3', color='red')


    # 3D投影
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    w2_dataset = data.loc[data['y'] == 2, 'x1':'x3'].to_numpy()
    w3_dataset = data.loc[data['y'] == 3, 'x1':'x3'].to_numpy()

    # 绘制原始数据点
    ax.scatter(w2_dataset[:, 0], w2_dataset[:, 1], w2_dataset[:, 2],
               marker='o', label='类别-2', color='green')
    ax.scatter(w3_dataset[:, 0], w3_dataset[:, 1], w3_dataset[:, 2],
               marker='^', label='类别-3', color='red')

    # 绘制投影后的点
    ax.scatter(w2_converted_dataset[:, 0],
               w2_converted_dataset[:, 1],
               w2_converted_dataset[:, 2],
               marker='o', label='投影后的类别-2',
               color='green')
    ax.scatter(w3_converted_dataset[:, 0],
               w3_converted_dataset[:, 1],
               w3_converted_dataset[:, 2],
               marker='^', label='投影后的类别-3',
               color='red')

    # 绘制最优w的直线
    plt.plot([w[0] * -2, w[0] * 2],
             [w[1] * -2, w[1] * 2],
             [w[2] * -2, w[2] * 2],
             label='非最优方向$w$的直线')

    # 绘制投影线
    for points in zip(w2_dataset, w2_converted_dataset):
        plt.plot([points[0][0], points[1][0]],
                 [points[0][1], points[1][1]],
                 [points[0][2], points[1][2]], color='green',
                 linestyle="--")
    for points in zip(w3_dataset, w3_converted_dataset):
        plt.plot([points[0][0], points[1][0]],
                 [points[0][1], points[1][1]],
                 [points[0][2], points[1][2]], color='red',
                 linestyle="--")

    ax.set_xlabel('X1 Label')
    ax.set_ylabel('X2 Label')
    ax.set_zlabel('X3 Label')

    plt.title('非最优方向的W和数据点投影')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    dataset = pd.read_csv('datasets/ex_1.csv')

    # # b题
    # program_2(dataset)
    # print('----------')
    # # c题
    # program_3(dataset)
    # print('----------')
    # # d题
    # program_4(dataset)
    # print('----------')
    # # e题
    # program_5(dataset)
    # # print('----------')
    # f题
    program_6(dataset)


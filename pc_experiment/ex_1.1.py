import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal, norm


def program_1(data):
    # 提取x1、x2、x3
    x1 = data.loc[data['y'] == 1, 'x1'].to_numpy()
    x2 = data.loc[data['y'] == 1, 'x2'].to_numpy()
    x3 = data.loc[data['y'] == 1, 'x3'].to_numpy()

    # 使用最大似然估计获得均值和方差
    mu_1, sigma_1 = norm.fit(x1)
    mu_2, sigma_2 = norm.fit(x2)
    mu_3, sigma_3 = norm.fit(x3)

    print('P127 Prb1 a题结果')
    print('w1类的x1的均值为{:.2f}，方差为{:.2f}'.format(mu_1, sigma_1**2))
    print('w1类的x2的均值为{:.2f}，方差为{:.2f}'.format(mu_2, sigma_2**2))
    print('w1类的x3的均值为{:.2f}，方差为{:.2f}'.format(mu_3, sigma_3**2))


def program_2(data):
    # 提取x1、x2、x3
    x1_x2 = data.loc[data['y'] == 1, ['x1', 'x2']].to_numpy()
    x1_x3 = data.loc[data['y'] == 1, ['x1', 'x3']].to_numpy()
    x2_x3 = data.loc[data['y'] == 1, ['x2', 'x3']].to_numpy()

    # 均值
    mu_12 = np.mean(x1_x2, axis=0)
    mu_13 = np.mean(x1_x3, axis=0)
    mu_23 = np.mean(x2_x3, axis=0)

    # 协方差
    # bias=True为True表示使用N作为归一化参数，rowvar=False表示每一列作为一个变量，每一行作为一个样本
    cov_12 = np.cov(x1_x2 - mu_12, bias=True, rowvar=False)
    cov_13 = np.cov(x1_x3 - mu_13, bias=True, rowvar=False)
    cov_23 = np.cov(x2_x3 - mu_23, bias=True, rowvar=False)

    print('P127 Prb1 b题结果')
    print('w1类的[x1, x2]的均值为{}，协方差为\n{}'.format(mu_12, cov_12))
    print('w1类的[x1, x3]的均值为{}，协方差为\n{}'.format(mu_13, cov_13))
    print('w1类的[x2, x3]的均值为{}，协方差为\n{}'.format(mu_23, cov_23))


def program_3(data):
    # 提取w1的x
    x = data.loc[data['y'] == 1, 'x1':'x3'].to_numpy()
    # 使用最大似然估计获得均值和协方差
    mu = np.mean(x, axis=0)
    cov = np.cov(x - mu, bias=True, rowvar=False)

    print('P127 Prb1 c题结果')
    print('w1类的[x1, x2, x3]的均值为{}，方差为\n{}'.format(mu, cov))


def program_4(data):
    # 提取w1的x
    x = data.loc[data['y'] == 2, 'x1':'x3'].to_numpy()
    # 使用最大似然估计获得均值和协方差
    mu = np.mean(x, axis=0)

    # 提取x1、x2、x3
    x1 = data.loc[data['y'] == 2, 'x1'].to_numpy()
    x2 = data.loc[data['y'] == 2, 'x2'].to_numpy()
    x3 = data.loc[data['y'] == 2, 'x3'].to_numpy()

    # 使用最大似然估计获得均值和方差
    cov = np.diag([norm.fit(x1)[1]**2, norm.fit(x2)[1]**2, norm.fit(x3)[1]**2])

    print('P127 Prb1 d题结果')
    print('w2类的[x1, x2, x3]的均值为{}，方差为\n{}'.format(mu, cov))


if __name__ == '__main__':
    dataset = pd.read_csv('datasets/ex_1.csv')

    # a题
    program_1(dataset)
    print('----------')
    # b题
    program_2(dataset)
    print('----------')
    # c题
    program_3(dataset)
    print('----------')
    # d题
    program_4(dataset)

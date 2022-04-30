import numpy as np
import matplotlib.pyplot as plt
import random


class LinearClassifier:
    def __init__(self, weights=None, lr=0.001, max_iter=1000):
        self.weights = weights
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, X, y):
        for i in range(self.max_iter):
            gradient = 2 * np.dot(np.transpose(X), np.dot(X, self.weights) -y)
            self.weights = self.weights - self.lr * gradient

        return self

    def score(self, X, y):
        # 预测的Y值
        y_hat = np.dot(X, self.weights)

        # 计算MSE
        return np.sum(np.power((y_hat - y), 2), axis=0)


def gauss_noisy(mu=0, sigma=1.0):
    return random.gauss(mu, sigma)


def target_function(x):
    return x ** 2


def noisy_target_function(x):
    return target_function(x) + gauss_noisy(0, 0.1)


def build_dataset(gx_type, samples_size=10):
    X = np.random.uniform(-1, 1, samples_size)
    y = noisy_target_function(X)

    if gx_type == 2:
        X = np.insert(X.reshape(-1, 1), 0, values=1, axis=1)
    elif gx_type == 3:
        _X = X.reshape(-1, 1)
        X = np.hstack((np.hstack((np.insert(_X, 0, values=1, axis=1),
                                  np.power(_X, 2))), np.power(_X, 3)))

    return X, y


def deviation_variance(g_x, F_x):
    deviation = np.power(np.mean(g_x - F_x.reshape(-1, 1), axis=0), 2)
    variance = np.var(g_x)

    return deviation, variance, deviation + variance


def constant_regression(gx_type=0):
    if gx_type == 0:
        return 0.5
    elif gx_type == 1:
        return 1.0
    else:
        return 0.5


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    hist_deviation_0 = []
    hist_variance_0 = []
    hist_deviation_variance_0 = []

    hist_deviation_1 = []
    hist_variance_1 = []
    hist_deviation_variance_1 = []

    hist_deviation_2 = []
    hist_variance_2 = []
    hist_deviation_variance_2 = []

    hist_deviation_3 = []
    hist_variance_3 = []
    hist_deviation_variance_3 = []

    # 每个数据集的样本数
    samples_size = 100

    for i in range(100):
        X_0, y_0 = build_dataset(gx_type=0, samples_size=samples_size)
        _deviation_0, _variance_0, _deviation_variance_0 = deviation_variance(constant_regression(0), y_0)
        hist_deviation_0.append(_deviation_0)
        hist_variance_0.append(_variance_0)
        hist_deviation_variance_0.append(_deviation_variance_0)

        X_1, y_1 = build_dataset(gx_type=1, samples_size=samples_size)
        _deviation_1, _variance_1, _deviation_variance_1 = deviation_variance(constant_regression(1), y_1)
        hist_deviation_1.append(_deviation_1)
        hist_variance_1.append(_variance_1)
        hist_deviation_variance_1.append(_deviation_variance_1)

        X_2, y_2 = build_dataset(gx_type=2, samples_size=samples_size)
        model_2 = LinearClassifier(weights=np.random.randn(
            X_2.shape[1]).reshape(-1, 1)).fit(X_2, y_2.reshape(-1, 1))
        y_hat_2 = np.dot(X_2, model_2.weights)
        _deviation_2, _variance_2, _deviation_variance_2 = deviation_variance(y_hat_2, y_2)
        hist_deviation_2.append(_deviation_2)
        hist_variance_2.append(_variance_2)
        hist_deviation_variance_2.append(_deviation_variance_2)

        X_3, y_3 = build_dataset(gx_type=3, samples_size=samples_size)
        model_3 = LinearClassifier(weights=np.random.randn(
            X_3.shape[1]).reshape(-1, 1)).fit(X_3, y_3.reshape(-1, 1))
        y_hat_3 = np.dot(X_3, model_3.weights)
        _deviation_3, _variance_3, _deviation_variance_3 = deviation_variance(y_hat_3, y_3)
        hist_deviation_3.append(_deviation_3)
        hist_variance_3.append(_variance_3)
        hist_deviation_variance_3.append(_deviation_variance_3)

    plt.subplot(4, 3, 1)
    hist_deviation_0 = np.array(hist_deviation_0).flatten()
    plt.hist(np.arange(len(hist_deviation_0)), bins=len(hist_deviation_0),
             weights=hist_deviation_0, label='回归函数$g(x)=0.5$的偏差')
    plt.xlabel('数据集序编号')
    plt.ylabel('偏差')
    plt.grid(True)
    plt.legend()
    plt.subplot(4, 3, 2)
    hist_variance_0 = np.array(hist_variance_0).flatten()
    plt.hist(np.arange(len(hist_variance_0)), bins=len(hist_variance_0),
             weights=hist_variance_0, label='回归函数$g(x)=0.5$的方差')
    plt.xlabel('数据集序编号')
    plt.ylabel('方差')
    plt.grid(True)
    plt.legend()
    plt.subplot(4, 3, 3)
    hist_deviation_variance_0 = np.array(hist_deviation_variance_0).flatten()
    plt.hist(np.arange(len(hist_deviation_variance_0)), bins=len(hist_deviation_variance_0),
             weights=hist_deviation_variance_0, label='回归函数$g(x)=0.5$的偏差、方差和')
    plt.xlabel('数据集序编号')
    plt.ylabel('偏差、方差的和')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 3, 4)
    hist_deviation_1 = np.array(hist_deviation_1).flatten()
    plt.hist(np.arange(len(hist_deviation_1)), bins=len(hist_deviation_1),
             weights=hist_deviation_1, label='回归函数$g(x)=1.0$的偏差')
    plt.xlabel('数据集序编号')
    plt.ylabel('偏差')
    plt.grid(True)
    plt.legend()
    plt.subplot(4, 3, 5)
    hist_variance_1 = np.array(hist_variance_1).flatten()
    plt.hist(np.arange(len(hist_variance_1)), bins=len(hist_variance_1),
             weights=hist_variance_1, label='回归函数$g(x)=1.0$的方差')
    plt.xlabel('数据集序编号')
    plt.ylabel('方差')
    plt.grid(True)
    plt.legend()
    plt.subplot(4, 3, 6)
    hist_deviation_variance_1 = np.array(hist_deviation_variance_1).flatten()
    plt.hist(np.arange(len(hist_deviation_variance_1)), bins=len(hist_deviation_variance_1),
             weights=hist_deviation_variance_1, label='回归函数$g(x)=1.0$的偏差、方差和')
    plt.xlabel('数据集序编号')
    plt.ylabel('偏差、方差的和')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 3, 7)
    hist_deviation_2 = np.array(hist_deviation_2).flatten()
    plt.hist(np.arange(len(hist_deviation_2)), bins=len(hist_deviation_2),
             weights=hist_deviation_2, label='回归函数$g(x)=a_{0} + a_{1} \cdot x$的偏差')
    plt.xlabel('数据集序编号')
    plt.ylabel('偏差')
    plt.grid(True)
    plt.legend()
    plt.subplot(4, 3, 8)
    hist_variance_2 = np.array(hist_variance_2).flatten()
    plt.hist(np.arange(len(hist_variance_2)), bins=len(hist_variance_2),
             weights=hist_variance_2, label='回归函数$g(x)=a_{0} + a_{1} \cdot x$的方差')
    plt.xlabel('数据集序编号')
    plt.ylabel('方差')
    plt.grid(True)
    plt.legend()
    plt.subplot(4, 3, 9)
    hist_deviation_variance_2 = np.array(hist_deviation_variance_2).flatten()
    plt.hist(np.arange(len(hist_deviation_variance_2)), bins=len(hist_deviation_variance_2),
             weights=hist_deviation_variance_2, label='回归函数$g(x)=a_{0} + a_{1} \cdot x$的偏差、方差和')
    plt.xlabel('数据集序编号')
    plt.ylabel('偏差、方差的和')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 3, 10)
    hist_deviation_3 = np.array(hist_deviation_3).flatten()
    plt.hist(np.arange(len(hist_deviation_3)), bins=len(hist_deviation_3),
             weights=hist_deviation_3,
             label='回归函数$g(x)=a_{0} + a_{1} \cdot x + a_{2} \cdot x^{2} + a_{3} \cdot x^{3}$的偏差')
    plt.xlabel('数据集序编号')
    plt.ylabel('偏差')
    plt.grid(True)
    plt.legend()
    plt.subplot(4, 3, 11)
    hist_variance_3 = np.array(hist_variance_3).flatten()
    plt.hist(np.arange(len(hist_variance_3)), bins=len(hist_variance_3),
             weights=hist_variance_3,
             label='回归函数$g(x)=a_{0} + a_{1} \cdot x + a_{2} \cdot x^{2} + a_{3} \cdot x^{3}$的方差')
    plt.xlabel('数据集序编号')
    plt.ylabel('方差')
    plt.grid(True)
    plt.legend()
    plt.subplot(4, 3, 12)
    hist_deviation_variance_3 = np.array(hist_deviation_variance_3).flatten()
    plt.hist(np.arange(len(hist_deviation_variance_3)), bins=len(hist_deviation_variance_3),
             weights=hist_deviation_variance_3,
             label='回归函数$g(x)=a_{0} + a_{1} \cdot x + a_{2} \cdot x^{2} + a_{3} \cdot x^{3}$的偏差、方差和')
    plt.xlabel('数据集序编号')
    plt.ylabel('偏差、方差的和')
    plt.grid(True)
    plt.legend()

    plt.suptitle('样本数 = {} 时的偏差、方差，以及偏差、方差和'.format(samples_size))
    plt.subplots_adjust(top=0.92, bottom=0.06, hspace=0.275)
    plt.show()

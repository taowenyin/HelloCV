import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


def bgd(X, Y, tau=0.001, alpha=0.001, max_itr=1000,
        theta_init=[], diagnosis=True):
    # initialize
    counter = 0

    # theta denotes coefficients
    # initialize theta
    if len(theta_init) > 0:
        theta = theta_init
    else:
        theta = np.ones((X.shape[1], 1))

    # residuals or you can call it objective function
    epsilons = []

    # collect thetas for viz
    thetas = [theta.copy()]

    # iterate through the entire dataset
    while counter < max_itr:
        x = X.copy()
        y = Y.copy()

        # gradient descent
        y_estimated = x @ theta
        epsilon = y_estimated - y
        gradient = x.T @ epsilon
        theta -= alpha * gradient

        # tracking
        counter += 1
        epsilons.append(epsilon.ravel()[0])
        thetas.append(theta.copy())

        # convergence
        if len(epsilons) > 1 and abs(epsilons[-1] - epsilons[-2]) < tau:
            if diagnosis:
                print(f'Converged after {counter} iterations.')
            return thetas, epsilons

    if diagnosis:
        print(f'Not converged.')
    return thetas, epsilons


def dataset_random_choice(data, target, category):
    # 提取指定类别的数据和标签
    class0_data = data[target == category[0]]
    class0_target = target[target == category[0]]

    class1_data = data[target == category[1]]
    class1_target = target[target == category[1]]

    # 构建训练样本的索引
    train_index = np.arange(len(class0_target))

    # 随机选取25个训练样本的索引
    class0_train_index = np.random.choice(train_index, 25, replace=False)
    # 提取剩余的索引作为测试样本
    class0_test_index = np.delete(train_index, class0_train_index, axis=0)
    # 根据训练和测试样本的索引提取对应的数据和标签
    class0_train_X = class0_data[class0_train_index]
    class0_test_X = class0_data[class0_test_index]
    class0_train_y = class0_target[class0_train_index]
    class0_test_y = class0_target[class0_test_index]

    class1_train_index = np.random.choice(train_index, 25, replace=False)
    class1_test_index = np.delete(train_index, class1_train_index, axis=0)
    class1_train_X = class1_data[class1_train_index]
    class1_test_X = class1_data[class1_test_index]
    class1_train_y = class1_target[class1_train_index]
    class1_test_y = class1_target[class1_test_index]

    # 组合数组和标签，并进行打乱
    train_X = np.vstack((class0_train_X, class1_train_X))
    train_y = np.append(class0_train_y, class1_train_y)
    train = np.hstack((train_X, train_y.reshape(-1, 1)))
    np.random.shuffle(train)
    # 把打乱的数据再分离为数据和标签
    train_X = np.array(train[:, :-1])
    train_y = np.array(train[:, -1], dtype=np.int64)

    test_X = np.vstack((class0_test_X, class1_test_X))
    test_y = np.append(class0_test_y, class1_test_y)
    test = np.hstack((test_X, test_y.reshape(-1, 1)))
    np.random.shuffle(test)
    test_X = np.array(test[:, :-1])
    test_y = np.array(test[:, -1], dtype=np.int64)

    return train_X, test_X, train_y, test_y

if __name__ == '__main__':
    # 加载鸢尾花数据集的特征和类标签
    iris = datasets.load_iris()
    # dataset = iris.data[iris.target != 2]
    # Y = iris.target[iris.target != 2]

    _target = iris.target
    # 数据做归一化
    _data = (iris.data - np.mean(iris.data, axis=0)) / np.std(iris.data, axis=0)

    train_X, test_X, train_y, test_y = dataset_random_choice(_data, _target, [0, 2])

    # X_2 = PCA(n_components=4).fit_transform(dataset)
    # X = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
    # Y = Y.reshape(-1, 1)

    constant = np.ones((train_X.shape[0], 1))
    X = np.concatenate([constant, train_X], axis=1)

    thetas_bgd, _ = bgd(X, train_y.reshape(-1, 1))
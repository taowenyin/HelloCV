import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn import datasets
from collections import Counter


def dataset_random_choice(data, target, category, data_enlarge=True):
    # 对数据进行增广
    if data_enlarge:
        data = np.insert(data, 0, values=1, axis=1)

    # 提取指定类别的数据和标签
    class0_data = data[target == category[0]]
    class1_data = data[target == category[1]]

    # class0_target = np.ones(target[target == category[0]].shape)
    # class1_target = np.full(target[target == category[1]].shape, -1)
    class0_target = target[target == category[0]]
    class1_target = target[target == category[1]]
    # class0_target = np.zeros(target[target == category[0]].shape)
    # class1_target = np.ones(target[target == category[1]].shape)

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
    # # 按照P183线性可分的规范化操作，要把训练集的第二类数据X乘-1，使得得到图5-8右边的效果，而测试集不需要
    class1_train_X = -1 * class1_data[class1_train_index]
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
    iris = datasets.load_iris()
    _target = iris.target
    # 数据做归一化
    _data = (iris.data - np.mean(iris.data, axis=0)) / np.std(iris.data, axis=0)

    train_X, test_X, train_y, test_y = dataset_random_choice(_data, _target, [0, 2])

    # class0_data = _data[_target == 0]
    # class1_data = -1 * _data[_target == 1]
    # 按照P183线性可分的规范化操作，把Class1类的标签设置为1，Class2类的标签设置为-1
    # class0_target = np.ones(_target[_target == 0].shape)
    # class1_target = np.full(_target[_target == 2].shape, -1)


    # X = np.vstack((class0_data, class1_data))
    # trn_lable = np.append(class0_target, class1_target)
    w = np.random.randn(train_X.shape[1], 1)
    b = np.random.rand(train_X.shape[0], 1)
    eta = 0.1
    MAX_iteration = 1000

    step = 0
    acc = 0
    for i in range(MAX_iteration):
        e = np.dot(train_X, w) - b
        e_p = 0.5 * (e + abs(e))
        eta_k = eta/(i+1)
        # if (e == 0).all():
        if max(abs(e)) <= min(b):
            print(e)
            break
        else:
            b += 2 * eta_k * e_p
            Y_p = np.linalg.pinv(train_X)
            w = np.dot(Y_p, b)
        step += 1
        # acc = (np.sum(np.dot(train_X, w) > 0)) / train_X.shape[0]
        # print('第%2d次更新, 分类准确率%f' % (step, acc))
        # # print(e, '\n')
        # if step == MAX_iteration and acc != 1:
        #     print("未找到解，线性不可分！")

    y_hat = np.dot(test_X, w)

    y_label = np.sort(np.unique(test_y))
    y_hat[y_hat > 0] = y_label[0]
    y_hat[y_hat < 0] = y_label[1]
    y_hat = np.array(y_hat.reshape(1, -1)[0], dtype=np.int64)

    accuracy_hk_perception = np.sum((y_hat == test_y)) / len(test_y)

    print(accuracy_hk_perception)
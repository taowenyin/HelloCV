import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from collections import Counter


class MSELoss:
    # 使用MSE作为损失函数
    def loss(self, X, y, weights):
        # 样本大小
        samples_size = len(X)

        # 计算差值
        residual = np.dot(X, weights) - y
        # 计算损失
        loss = 1 / (2 * samples_size) * np.dot(np.transpose(residual), residual)

        return loss

    # 损失函数求导
    def loss_gradient(self, X, y, weights):
        # 样本大小
        samples_size = len(X)

        # 计算残差
        residual = np.dot(X, weights) - y
        # 计算梯度
        gradient = (1 / samples_size) * np.dot(np.transpose(X), residual)

        return gradient, residual


class Optimizer:
    """
    梯度下降法
    """
    def __init__(self, loss_func: MSELoss, method, weights=None,
                 threshold=1e-5, margin=1.0, winnow_alpha=2):
        self.loss_func = loss_func
        self.method = method
        self.weights = weights
        self.winnow_alpha = winnow_alpha

        # 记录每次的梯度值、方差值、损失
        self.gradient_collection = None
        self.residual_collection = []
        self.loss_collection = []

        # 带Margin的变增量感知机算法中的Margin参数
        self.threshold = threshold
        self.margin = -1 * margin

    def optimize(self, X, y, lr, max_iter):
        # 迭代的次数
        step = 0
        # Perception算法中用于记录错分类的个数
        error_count = 0

        while step < max_iter:
            # 计算梯度
            gradient, residual = self.loss_func.loss_gradient(X, y, self.weights)
            # 每次迭代初始化
            if (self.method != 'OneSamplePerception') and (self.method != 'MarginPerception'):
                error_count = 0

            # todo 算法添加
            if self.method == 'GD':
                # 参考书上P184，算法1：梯度下降公式
                threshold_gradient = lr * gradient
            elif self.method == 'Newtons':
                # 参考书上P185，算法2：牛顿下降公式
                H = np.linalg.pinv(2 * np.dot(np.transpose(X), X))
                threshold_gradient = np.dot(H, gradient)
            elif self.method == 'Perception':
                threshold_gradient = np.zeros(self.weights.shape)
                # 参考书上P186，算法3：感知机算法
                for i in range(len(X)):
                    data = X[i].reshape(1, -1)
                    target = y[i][0]
                    # 判断错分的样本
                    if target * np.dot(data, self.weights) <= 0:
                        threshold_gradient += lr * np.dot(target, data).reshape(-1, 1)
                        # 错分类+1
                        error_count += 1

                # 参考书上P186，公式17，需要在前面加个负号
                threshold_gradient = -1 * threshold_gradient
            elif self.method == 'OneSamplePerception':
                threshold_gradient = np.zeros(self.weights.shape)
                i = step % len(X)
                data = X[i].reshape(1, -1)
                target = y[i][0]
                # 参考书上P188，算法4：固定增量单样本感知机
                if target * np.dot(data, self.weights) <= 0:
                    # 注意：单样本是不需要乘以学习率
                    threshold_gradient = np.dot(target, data).reshape(-1, 1)
                    error_count += 1

                threshold_gradient = -1 * threshold_gradient
            elif self.method == 'MarginPerception':
                threshold_gradient = np.zeros(self.weights.shape)
                # 参考书上P191，学习率的计算
                lr = 1 / (step + 1)
                i = step % len(X)
                data = X[i].reshape(1, -1)
                target = y[i][0]
                # 参考书上P190，算法5：带Margin的变增量感知机
                if target * np.dot(data, self.weights) <= self.margin:
                    # 注意：带Margin的变增量感知机需要乘以学习率
                    threshold_gradient = lr * np.dot(target, data).reshape(-1, 1)
                    error_count += 1

                threshold_gradient = -1 * threshold_gradient
            elif self.method == 'BatchLrPerception':
                threshold_gradient = np.zeros(self.weights.shape)
                # 参考书上P191，学习率的计算
                lr = 1 / (step + 1)
                # 参考书上P191，算法6：批量变增量感知机算法
                for i in range(len(X)):
                    data = X[i].reshape(1, -1)
                    target = y[i][0]
                    # 判断错分的样本
                    if target * np.dot(data, self.weights) <= 0:
                        threshold_gradient += np.dot(target, data).reshape(-1, 1)
                        # 错分类+1
                        error_count += 1

                # 参考书上P186，公式17，需要在前面加个负号
                threshold_gradient = -1 * lr * threshold_gradient
            else:
                # 参考书上P184，算法1：梯度下降公式
                threshold_gradient = lr * gradient

            step += 1

            # 权重更新
            self.weights -= threshold_gradient

            # 保存平均的均方差和梯度
            if self.gradient_collection is None:
                self.gradient_collection = gradient.reshape(1, -1)
            else:
                self.gradient_collection = np.vstack((self.gradient_collection, gradient.reshape(1, -1)))

            self.residual_collection.append(residual.ravel()[0])

            # 计算并保存损失
            loss = self.loss_func.loss(X, y, self.weights)[0][0]
            self.loss_collection.append(loss)

            # 感知机的停止条件与其他不同
            if self.method == 'Perception':
                if error_count == 0:
                    print('经过{}次迭代{}收敛，训练错误率 = 0'.format(step, self.method))
                    return
            elif self.method == 'GD' or self.method == 'Newtons':
                # 如果梯度足够小，则停止计算
                # 书上的停止条件
                if all(abs(threshold_gradient) <= self.threshold):
                # if len(self.residual_collection) > 1 and \
                #         abs(self.residual_collection[-1] - self.residual_collection[-2] < self.threshold):
                    print('经过{}次迭代{}收敛，残差 = {:.2f}，损失 = {:.2f}'.format(
                        step, self.method, self.residual_collection[-1], loss))
                    return

        # 打印平均的梯度和均方差
        if self.method == 'Perception' or \
                self.method == 'OneSamplePerception' or \
                self.method == 'MarginPerception' or \
                self.method == 'BatchLrPerception' or \
                self.method == 'WinnowPerception':
            print('经过{}次迭代{}收敛，训练错误率 = {:.2f}'.format(step, self.method, error_count / len(X)))
        else:
            print('经过{}次迭代{}收敛，残差 = {:.2f}，损失 = {:.2f}'.format(
                step, self.method, self.residual_collection[-1], self.loss_collection[-1]))


class LinearClassifier:
    def __init__(self, optimal, lr=0.001, max_iter=1000):
        self.optimal = optimal
        self.lr = lr
        self.max_iter = max_iter

    def data_enlarge(self, data):
        if len(data.shape) > 1:
            data = np.insert(data, 0, values=1, axis=1)
        else:
            data = np.insert(data, 0, values=1)

        return data

    def fit(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)

        size, dim = self.X.shape
        # 使用广义线性判别，P182，公式10、11，要给x和weights增加一个维度，并且值为1
        self.X = self.data_enlarge(self.X)
        if self.optimal.weights is None:
            self.optimal.weights = np.ones(dim + 1).reshape(-1, 1)
        else:
            self.optimal.weights = self.data_enlarge(self.optimal.weights)

        self.optimal.optimize(self.X, self.y, self.lr, self.max_iter)

        return self

    def predict(self, X):
        X = self.data_enlarge(X)
        # 预测结果
        y = np.dot(X, self.optimal.weights)

        # 获取标签
        y_label = np.sort(np.unique(self.y))
        # 设置标签
        y[y > 0] = y_label[1]
        y[y < 0] = y_label[0]
        y = np.array(y.reshape(1, -1)[0], dtype=np.int64)

        return y

    def score(self, X, y):
        y = y.reshape(-1, 1)
        X = self.data_enlarge(X)

        variance = self.optimal.loss_func.loss(X, y, self.optimal.weights)
        return variance[0][0]


def dataset_random_choice(data, target, category):
    # 提取指定类别的数据和标签
    class0_data = data[target == category[0]]
    class1_data = data[target == category[1]]
    # 按照P183线性可分的规范化操作，把Class1类的标签设置为1，Class2类的标签设置为-1
    class0_target = np.ones(target[target == category[0]].shape)
    class1_target = np.full(target[target == category[1]].shape, -1)

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


def program_1(dataset):
    _target = dataset.target
    # 数据做归一化
    _data = (dataset.data - np.mean(dataset.data, axis=0)) / np.std(dataset.data, axis=0)
    # 迭代次数
    epoch = 100

    # todo 保存精度
    gd_accuracy_collection = []
    newtons_accuracy_collection = []
    perception_accuracy_collection = []
    one_sample_perception_accuracy_collection = []
    margin_perception_accuracy_collection = []
    batch_lr_perception_accuracy_collection = []
    winnow_perception_accuracy_collection = []

    for i in range(epoch):
        # 对数据集随机采样，并构建训练集和测试集数据
        train_X, test_X, train_y, test_y = dataset_random_choice(_data, _target, [0, 2])

        # 定义MSE损失函数
        loss = MSELoss()

        print('==========周期{}=========='.format(i + 1))

        # todo 算法使用
        # ==========================算法1：梯度下降算法=======================
        # 使用GD优化器进行优化
        optimal_gd = Optimizer(loss, method='GD')
        # 使用相同的训练流程，但选择不同的优化器来拟合训练数据
        model_gd = LinearClassifier(optimal_gd, lr=0.001).fit(train_X, train_y)
        # 预测结果
        test_predict_gd = model_gd.predict(test_X)
        # 计算精度
        accuracy_gd = Counter(np.multiply(test_predict_gd, test_y))[1] / len(test_y)
        # 保存精度
        gd_accuracy_collection.append(accuracy_gd)

        # ==========================算法2：牛顿下降算法=======================
        optimal_newtons = Optimizer(loss, method='Newtons')
        model_newtons = LinearClassifier(optimal_newtons, lr=0.001).fit(train_X, train_y)
        test_predict_newtons = model_newtons.predict(test_X)
        accuracy_newtons = Counter(np.multiply(test_predict_newtons, test_y))[1] / len(test_y)
        newtons_accuracy_collection.append(accuracy_newtons)

        # ==========================算法3：感知机准则算法=======================
        optimal_perception = Optimizer(loss, method='Perception')
        model_perception = LinearClassifier(optimal_perception, lr=0.001).fit(train_X, train_y)
        test_predict_perception = model_perception.predict(test_X)
        accuracy_perception = Counter(np.multiply(test_predict_perception, test_y))[1] / len(test_y)
        perception_accuracy_collection.append(accuracy_perception)

        # ==========================算法4：单样本感知机准则算法=======================
        optimal_one_sample_perception = Optimizer(loss, method='OneSamplePerception')
        model_one_sample_perception = LinearClassifier(optimal_one_sample_perception,
                                                       lr=0.001).fit(train_X, train_y)
        test_predict_one_sample_perception = model_one_sample_perception.predict(test_X)
        accuracy_one_sample_perception = Counter(np.multiply(test_predict_one_sample_perception,
                                                             test_y))[1] / len(test_y)
        one_sample_perception_accuracy_collection.append(accuracy_one_sample_perception)

        # ==========================算法5：带间隔感知机准则算法=======================
        optimal_margin_perception = Optimizer(loss, method='MarginPerception', margin=0.1)
        model_margin_perception = LinearClassifier(optimal_margin_perception,
                                                   lr=0.001).fit(train_X, train_y)
        test_predict_margin_perception = model_margin_perception.predict(test_X)
        accuracy_margin_perception = Counter(np.multiply(test_predict_margin_perception,
                                                         test_y))[1] / len(test_y)
        margin_perception_accuracy_collection.append(accuracy_margin_perception)

        # ==========================算法6：批量变增量感知机准则算法=======================
        optimal_batch_lr_perception = Optimizer(loss, method='BatchLrPerception', margin=0.1)
        model_batch_lr_perception = LinearClassifier(optimal_batch_lr_perception,
                                                   lr=0.001).fit(train_X, train_y)
        test_predict_batch_lr_perception = model_batch_lr_perception.predict(test_X)
        accuracy_batch_lr_perception = Counter(np.multiply(test_predict_batch_lr_perception,
                                                           test_y))[1] / len(test_y)
        batch_lr_perception_accuracy_collection.append(accuracy_batch_lr_perception)

        # # ==========================算法7：Winnow算法=======================
        # optimal_winnow_perception = Optimizer(loss, method='WinnowPerception', margin=0.1)
        # model_winnow_perception = LinearClassifier(optimal_winnow_perception,
        #                                            lr=0.001).fit(train_X, train_y)
        # test_predict_winnow_perception = model_winnow_perception.predict(test_X)
        # accuracy_winnow_perception = Counter(np.multiply(test_predict_winnow_perception,
        #                                                  test_y))[1] / len(test_y)
        # winnow_perception_accuracy_collection.append(accuracy_winnow_perception)

    # 画图
    epochs = range(1, epoch + 1)

    # todo 绘图
    plt.plot(epochs, gd_accuracy_collection,
             label='算法1：梯度下降平均精度={:.4f}和精度方差{:.4f}'.format(
                 np.mean(gd_accuracy_collection), np.var(gd_accuracy_collection)))
    plt.plot(epochs, newtons_accuracy_collection,
             label='算法2：牛顿平均精度={:.4f}和精度方差{:.4f}'.format(
                 np.mean(newtons_accuracy_collection), np.var(newtons_accuracy_collection)))
    plt.plot(epochs, perception_accuracy_collection,
             label='算法3：感知机平均精度={:.4f}和精度方差{:.4f}'.format(
                 np.mean(perception_accuracy_collection), np.var(perception_accuracy_collection)))
    plt.plot(epochs, one_sample_perception_accuracy_collection,
             label='算法4：单样本感知机平均精度={:.4f}和精度方差{:.4f}'.format(
                 np.mean(one_sample_perception_accuracy_collection),
                 np.var(one_sample_perception_accuracy_collection)))
    plt.plot(epochs, margin_perception_accuracy_collection,
             label='算法5：带间隔感知机平均精度={:.4f}和精度方差{:.4f}'.format(
                 np.mean(margin_perception_accuracy_collection),
                 np.var(margin_perception_accuracy_collection)))
    plt.plot(epochs, batch_lr_perception_accuracy_collection,
             label='算法6：批变增量感知机平均精度={:.4f}和精度方差{:.4f}'.format(
                 np.mean(batch_lr_perception_accuracy_collection),
                 np.var(batch_lr_perception_accuracy_collection)))
    # plt.plot(epochs, winnow_perception_accuracy_collection,
    #          label='算法7：Winnow算法平均精度={:.4f}和精度方差{:.4f}'.format(
    #              np.mean(winnow_perception_accuracy_collection),
    #              np.var(winnow_perception_accuracy_collection)))

    plt.title('不同优化器的平均精度和精度的方差')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    iris = datasets.load_iris()

    # a题
    program_1(iris)
    print('----------')


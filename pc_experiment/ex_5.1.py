import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from tqdm import tqdm


class Optimizer:
    """
    梯度下降法
    """
    def __init__(self, method, weights=None, threshold=1e-5,
                 hk_b=None, margin=1.0, winnow_alpha=2):
        self.method = method
        self.weights = weights
        self.winnow_alpha = winnow_alpha

        # 记录每次的梯度值、方差值、损失
        self.gradient_collection = None

        # 带Margin的变增量感知机算法中的Margin参数
        self.threshold = threshold
        self.margin = margin
        self.hk_b = hk_b

    def optimize(self, X, y, lr, max_iter):
        # 迭代的次数
        step = 0
        # Perception算法中用于记录错分类的个数
        error_count = 0

        while step < max_iter:
            # 每次迭代初始化
            if (self.method != 'OneSamplePerception') and (self.method != 'MarginPerception'):
                error_count = 0

            # todo 算法添加
            if self.method == 'GD':
                # 参考书上P184，算法1：梯度下降公式，并使用P186的感知机准则函数的梯度公式（17）
                gradient = np.sum(X, axis=0).reshape(-1, 1)
                threshold_gradient = lr * gradient

                # 参考书上P186，公式17，需要在前面加个负号
                threshold_gradient = -1 * threshold_gradient
            elif self.method == 'Newtons':
                # 参考书上P185，算法2：牛顿下降公式，并使用P186的感知机准则函数的梯度公式（17）
                gradient = np.sum(X, axis=0).reshape(-1, 1)
                H = np.linalg.pinv(2 * np.dot(np.transpose(X), X))
                threshold_gradient = np.dot(H, gradient)

                # 参考书上P186，公式17，需要在前面加个负号
                threshold_gradient = -1 * threshold_gradient
            elif self.method == 'Perception':
                threshold_gradient = np.zeros(self.weights.shape)
                # 参考书上P186，算法3：感知机算法
                for i in range(len(X)):
                    data = X[i].reshape(1, -1)
                    target = y[i][0]
                    # 判断错分的样本
                    if np.dot(data, self.weights) <= 0:
                        threshold_gradient += lr * data.reshape(-1, 1)
                        # 错分类+1
                        error_count += 1

                # 参考书上P186，公式17，需要在前面加个负号
                threshold_gradient = -1 * threshold_gradient
            elif self.method == 'OneSamplePerception':
                threshold_gradient = np.zeros(self.weights.shape)
                i = step % len(X)
                data = X[i].reshape(1, -1)
                # 参考书上P188，算法4：固定增量单样本感知机
                if np.dot(data, self.weights) <= 0:
                    # 注意：单样本是不需要乘以学习率
                    threshold_gradient = data.reshape(-1, 1)
                    error_count += 1

                threshold_gradient = -1 * threshold_gradient
            elif self.method == 'MarginPerception':
                threshold_gradient = np.zeros(self.weights.shape)
                # 参考书上P191，学习率的计算
                lr = 1 / (step + 1)
                i = step % len(X)
                data = X[i].reshape(1, -1)
                # 参考书上P190，算法5：带Margin的变增量感知机
                if np.dot(data, self.weights) <= self.margin:
                    # 注意：带Margin的变增量感知机需要乘以学习率
                    threshold_gradient = lr * data.reshape(-1, 1)
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
                    if np.dot(data, self.weights) <= 0:
                        threshold_gradient += data.reshape(-1, 1)
                        # 错分类+1
                        error_count += 1

                # 参考书上P186，公式17，需要在前面加个负号
                threshold_gradient = -1 * lr * threshold_gradient
            elif self.method == 'RelaxationMarginPerception':
                threshold_gradient = np.zeros(self.weights.shape)
                i = step % len(X)
                data = X[i].reshape(1, -1)

                # 参考书上P193，算法9：带Margin的松弛感知机
                if np.dot(data, self.weights) <= self.margin:
                    factor = lr * ((self.margin - np.dot(data, self.weights)) /
                                   (np.linalg.norm(data, ord=2) ** 2))
                    threshold_gradient = factor * data.reshape(-1, 1)
                    error_count += 1

                threshold_gradient = -1 * threshold_gradient
            elif self.method == 'LMS':
                i = step % len(X)
                data = X[i].reshape(1, -1)

                # 参考书上P201，算法10：LMS算法
                factor = lr * (self.margin - np.dot(data, self.weights))
                threshold_gradient = -1 * factor * data.reshape(-1, 1)
            elif self.method == 'Ho-Kashyap':
                # 为统一操作创建，无用
                threshold_gradient = np.zeros(self.weights.shape)

                # 参考书上P204，算法11：Ho-Kashyap算法
                e = np.dot(X, self.weights) - self.hk_b # [sample_size, 1]
                e_p = 0.5 * (e + abs(e))  # [sample_size, 1]
                lr_k = lr / (step + 1)
                if max(abs(e)) > min(self.hk_b):
                    self.hk_b += 2 * lr_k * e_p
                    self.weights = np.dot(np.linalg.pinv(X), self.hk_b) # [feature_num, 1]
                else:
                    break
            elif self.method == 'Plus-Ho-Kashyap':
                # 为统一操作创建，无用
                threshold_gradient = np.zeros(self.weights.shape)

                # 参考书上P207，算法12：Ho-Kashyap的改进算法
                e = np.dot(X, self.weights) - self.hk_b # [sample_size, 1]
                e_p = 0.5 * (e + abs(e))  # [sample_size, 1]
                lr_k = lr / (step + 1)
                if max(abs(e)) > min(self.hk_b):
                    self.hk_b += 2 * lr_k * (e + abs(e))
                    self.weights = np.dot(np.linalg.pinv(X), self.hk_b) # [feature_num, 1]
                else:
                    break
            else:
                # 参考书上P184，算法1：梯度下降公式，并使用P186的感知机准则函数的梯度公式（17）
                gradient = np.sum(X, axis=0).reshape(-1, 1)
                # 参考书上P184，算法1：梯度下降公式
                threshold_gradient = lr * gradient

            step += 1

            # 权重更新
            self.weights -= threshold_gradient

            # # 感知机的停止条件与其他不同
            # if self.method == 'Perception':
            #     if error_count == 0:
            #         print('经过{}次迭代{}收敛，训练错误率 = 0'.format(step, self.method))
            #         return
            # elif self.method == 'GD' or self.method == 'Newtons' or self.method == 'LMS':
            #     # 如果梯度足够小，则停止计算
            #     if all(abs(threshold_gradient) <= self.threshold):
            #         print('经过{}次迭代{}收敛'.format(step, self.method))
            #         break

        # # 打印平均的梯度和均方差
        # if self.method == 'Perception' or \
        #         self.method == 'OneSamplePerception' or \
        #         self.method == 'MarginPerception' or \
        #         self.method == 'BatchLrPerception' or \
        #         self.method == 'WinnowPerception' or \
        #         self.method == 'RelaxationMarginPerception':
        #     print('经过{}次迭代{}收敛，训练错误率 = {:.2f}'.format(step, self.method, error_count / len(X)))
        # else:
        #     print('经过{}次迭代{}收敛'.format(step, self.method))


class LinearClassifier:
    def __init__(self, optimal, lr=0.001, max_iter=1000):
        self.optimal = optimal
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)

        if self.optimal.weights is None:
            self.optimal.weights = np.random.randn(self.X.shape[1]).reshape(-1, 1)

        self.optimal.optimize(self.X, self.y, self.lr, self.max_iter)

        return self

    def predict(self, X):
        # 预测结果
        y_hat = np.dot(X, self.optimal.weights)

        return y_hat

    def score(self, X, y):
        y_hat = self.predict(X)
        y_label = np.sort(np.unique(y))
        # 大于0属于第一类，小于0属于第二类
        y_hat[y_hat > 0] = y_label[0]
        y_hat[y_hat < 0] = y_label[1]

        y_hat = np.array(y_hat.reshape(1, -1)[0], dtype=np.int64)

        # 返回正确率
        return np.sum((y_hat == y)) / len(y)


def dataset_random_choice(data, target, category, data_enlarge=True):
    # 默认对数据进行增广，使用广义线性判别，P182，公式10、11，要给x和weights增加一个维度，并且值为1
    if data_enlarge:
        data = np.insert(data, 0, values=1, axis=1)

    # 提取指定类别的数据和标签
    class0_data = data[target == category[0]]
    class1_data = data[target == category[1]]

    class0_target = target[target == category[0]]
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


def algorithm_compare(dataset, category, ax):
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
    relaxation_margin_perception_accuracy_collection = []
    lms_perception_accuracy_collection = []
    hk_perception_accuracy_collection = []
    plus_hk_perception_accuracy_collection = []

    for i in tqdm(range(epoch), desc='类{}和类{}在不同优化器下的比较'.format(category[0], category[1])):
        # 对数据集随机采样，并构建训练集和测试集数据
        train_X, test_X, train_y, test_y = dataset_random_choice(_data, _target, category)

        # todo 算法使用
        # ==========================算法1：梯度下降算法=======================
        # 使用GD优化器进行优化
        optimal_gd = Optimizer(method='GD')
        # 使用相同的训练流程，但选择不同的优化器来拟合训练数据
        model_gd = LinearClassifier(optimal_gd, lr=0.1).fit(train_X, train_y)
        # 预测结果
        accuracy_gd = model_gd.score(test_X, test_y)
        # 保存精度
        gd_accuracy_collection.append(accuracy_gd)

        # ==========================算法2：牛顿下降算法=======================
        optimal_newtons = Optimizer(method='Newtons')
        model_newtons = LinearClassifier(optimal_newtons, lr=0.001).fit(train_X, train_y)
        accuracy_newtons = model_newtons.score(test_X, test_y)
        newtons_accuracy_collection.append(accuracy_newtons)

        # ==========================算法3：感知机准则算法=======================
        optimal_perception = Optimizer(method='Perception')
        model_perception = LinearClassifier(optimal_perception, lr=0.001).fit(train_X, train_y)
        accuracy_perception = model_perception.score(test_X, test_y)
        perception_accuracy_collection.append(accuracy_perception)

        # ==========================算法4：单样本感知机准则算法=======================
        optimal_one_sample_perception = Optimizer(method='OneSamplePerception')
        model_one_sample_perception = LinearClassifier(optimal_one_sample_perception,
                                                       lr=0.001).fit(train_X, train_y)
        accuracy_one_sample_perception = model_one_sample_perception.score(test_X, test_y)
        one_sample_perception_accuracy_collection.append(accuracy_one_sample_perception)

        # ==========================算法5：带间隔感知机准则算法=======================
        optimal_margin_perception = Optimizer(method='MarginPerception', margin=0.1)
        model_margin_perception = LinearClassifier(optimal_margin_perception,
                                                   lr=0.001).fit(train_X, train_y)
        accuracy_margin_perception = model_margin_perception.score(test_X, test_y)
        margin_perception_accuracy_collection.append(accuracy_margin_perception)

        # ==========================算法6：批量变增量感知机准则算法=======================
        optimal_batch_lr_perception = Optimizer(method='BatchLrPerception', margin=0.1)
        model_batch_lr_perception = LinearClassifier(optimal_batch_lr_perception,
                                                     lr=0.001).fit(train_X, train_y)
        accuracy_batch_lr_perception = model_batch_lr_perception.score(test_X, test_y)
        batch_lr_perception_accuracy_collection.append(accuracy_batch_lr_perception)

        # # ==========================算法7：Winnow算法=======================
        # optimal_winnow_perception = Optimizer(loss, method='WinnowPerception', margin=0.1)
        # model_winnow_perception = LinearClassifier(optimal_winnow_perception,
        #                                            lr=0.001).fit(train_X, train_y)
        # test_predict_winnow_perception = model_winnow_perception.predict(test_X)
        # accuracy_winnow_perception = Counter(np.multiply(test_predict_winnow_perception,
        #                                                  test_y))[1] / len(test_y)
        # winnow_perception_accuracy_collection.append(accuracy_winnow_perception)
        #
        # ==========================算法9：带间隔的松弛感知机算法=======================
        optimal_relaxation_margin_perception = Optimizer(method='RelaxationMarginPerception',
                                                         margin=10)
        model_relaxation_margin_perception = LinearClassifier(optimal_relaxation_margin_perception,
                                                              lr=0.001).fit(train_X, train_y)
        accuracy_relaxation_margin_perception = model_relaxation_margin_perception.score(test_X, test_y)
        relaxation_margin_perception_accuracy_collection.append(accuracy_relaxation_margin_perception)

        # ==========================算法10：LMS算法=======================
        optimal_lms_perception = Optimizer(method='LMS', margin=10)
        model_lms_perception = LinearClassifier(optimal_lms_perception,
                                                lr=0.001).fit(train_X, train_y)
        accuracy_lms_perception = model_lms_perception.score(test_X, test_y)
        lms_perception_accuracy_collection.append(accuracy_lms_perception)

        # ==========================算法11：Ho-Kashyap算法=======================
        # 初始化b值
        hk_b = np.random.rand(train_X.shape[0], 1)
        optimal_hk_perception = Optimizer(method='Ho-Kashyap', hk_b=hk_b)
        model_hk_perception = LinearClassifier(optimal_hk_perception,
                                               lr=0.1).fit(train_X, train_y)
        accuracy_hk_perception = model_hk_perception.score(test_X, test_y)
        hk_perception_accuracy_collection.append(accuracy_hk_perception)

        # ==========================算法12：Ho-Kashyap改进算法=======================
        # 初始化b值
        hk_b = np.random.rand(train_X.shape[0], 1)
        optimal_plus_hk_perception = Optimizer(method='Plus-Ho-Kashyap', hk_b=hk_b)
        model_plus_hk_perception = LinearClassifier(optimal_plus_hk_perception,
                                                    lr=0.1).fit(train_X, train_y)
        accuracy_plus_hk_perception = model_plus_hk_perception.score(test_X, test_y)
        plus_hk_perception_accuracy_collection.append(accuracy_plus_hk_perception)

    # 画图
    epochs = range(1, epoch + 1)

    # todo 绘图
    ax.plot(epochs, gd_accuracy_collection,
            label='算法1：梯度下降平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(gd_accuracy_collection), np.var(gd_accuracy_collection)))
    print('算法1：梯度下降平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(gd_accuracy_collection), np.var(gd_accuracy_collection)))
    ax.plot(epochs, newtons_accuracy_collection,
            label='算法2：牛顿平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(newtons_accuracy_collection), np.var(newtons_accuracy_collection)))
    print('算法2：牛顿平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(newtons_accuracy_collection), np.var(newtons_accuracy_collection)))
    ax.plot(epochs, perception_accuracy_collection,
            label='算法3：感知机平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(perception_accuracy_collection), np.var(perception_accuracy_collection)))
    print('算法3：感知机平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(perception_accuracy_collection), np.var(perception_accuracy_collection)))
    ax.plot(epochs, one_sample_perception_accuracy_collection,
            label='算法4：单样本感知机平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(one_sample_perception_accuracy_collection),
                np.var(one_sample_perception_accuracy_collection)))
    print('算法4：单样本感知机平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(one_sample_perception_accuracy_collection),
        np.var(one_sample_perception_accuracy_collection)))
    ax.plot(epochs, margin_perception_accuracy_collection,
            label='算法5：带间隔感知机平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(margin_perception_accuracy_collection),
                np.var(margin_perception_accuracy_collection)))
    print('算法5：带间隔感知机平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(margin_perception_accuracy_collection),
        np.var(margin_perception_accuracy_collection)))
    ax.plot(epochs, batch_lr_perception_accuracy_collection,
            label='算法6：批变增量感知机平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(batch_lr_perception_accuracy_collection),
                np.var(batch_lr_perception_accuracy_collection)))
    print('算法6：批变增量感知机平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(batch_lr_perception_accuracy_collection),
        np.var(batch_lr_perception_accuracy_collection)))
    # plt.plot(epochs, winnow_perception_accuracy_collection,
    #          label='算法7：Winnow算法平均精度={:.4f}和精度方差{:.4f}'.format(
    #              np.mean(winnow_perception_accuracy_collection),
    #              np.var(winnow_perception_accuracy_collection)))
    ax.plot(epochs, relaxation_margin_perception_accuracy_collection,
            label='算法9：带间隔的松弛感知机算法平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(relaxation_margin_perception_accuracy_collection),
                np.var(relaxation_margin_perception_accuracy_collection)))
    print('算法9：带间隔的松弛感知机算法平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(relaxation_margin_perception_accuracy_collection),
        np.var(relaxation_margin_perception_accuracy_collection)))
    ax.plot(epochs, lms_perception_accuracy_collection,
            label='算法10：LMS算法平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(lms_perception_accuracy_collection),
                np.var(lms_perception_accuracy_collection)))
    print('算法10：LMS算法平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(lms_perception_accuracy_collection),
        np.var(lms_perception_accuracy_collection)))
    ax.plot(epochs, hk_perception_accuracy_collection,
            label='算法11：Ho-Kashyap算法平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(hk_perception_accuracy_collection),
                np.var(hk_perception_accuracy_collection)))
    print('算法11：Ho-Kashyap算法平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(hk_perception_accuracy_collection),
        np.var(hk_perception_accuracy_collection)))
    ax.plot(epochs, plus_hk_perception_accuracy_collection,
            label='算法12：Ho-Kashyap改进算法平均精度={:.4f}和精度方差{:.4f}'.format(
                np.mean(plus_hk_perception_accuracy_collection),
                np.var(plus_hk_perception_accuracy_collection)))
    print('算法12：Ho-Kashyap改进算法平均精度={:.4f}和精度方差{:.4f}'.format(
        np.mean(plus_hk_perception_accuracy_collection),
        np.var(plus_hk_perception_accuracy_collection)))

    ax.set_title('类{}和类{}在不同优化器下的平均精度和精度的方差'.format(category[0], category[1]))
    ax.legend()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    iris = datasets.load_iris()

    # 创建一个2x1的画布
    fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')

    # a题
    algorithm_compare(iris, [0, 2], ax1)
    print('----------')
    algorithm_compare(iris, [1, 2], ax2)

    plt.show()


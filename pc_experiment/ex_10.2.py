import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter


class DistanceFunction:
    def __init__(self, distance_type, beta=None):
        self.distance_type = distance_type
        self.beta = beta

    def calculate_distance(self, x, y):
        if self.distance_type == 'Euclid':
            return np.linalg.norm(x - y, ord=2)
        if self.distance_type == 'Gaussian':
            if self.beta is None:
                raise Exception('beta没有初始化')
            euclid_square = np.power(np.linalg.norm(x - y, ord=2), 2)

            return 1 - np.exp(-1 * self.beta * euclid_square)


class KCluster:
    def __init__(self, cluster_type, center_points,
                 distance_function: DistanceFunction, b=0,
                 max_iter=1000, center_threshold=2, weight_threshold=0):
        # K-Means Fuzzy-K-Means
        self.cluster_type = cluster_type
        self.center_points = center_points
        self.distance_function = distance_function
        self.category_data = []
        self.y = []
        self.step = 1
        self.weight_assistant = None
        self.max_iter = max_iter
        self.b = b
        self.center_threshold = center_threshold
        self.weight_threshold = weight_threshold

    def weight_update(self, X):
        for x_i, x in enumerate(X):
            dist_all = 0
            # 计算某个点到所有中点的距离，并求和
            for c_i, center in enumerate(self.center_points):
                # 计算平方距离
                euclid_square = np.power(self.distance_function.calculate_distance(x, center), 2)

                if euclid_square != 0:
                    dist_all += np.power(euclid_square, (1 / (1 - self.b)))

            for c_i, center in enumerate(self.center_points):
                euclid_square = np.power(self.distance_function.calculate_distance(x, center), 2)
                # 更新权重
                if euclid_square != 0:
                    dist_self = np.power(euclid_square, (1 / (1 - self.b)))
                    self.weight_assistant[x_i, c_i + 2] = dist_self / dist_all
                else:
                    # 如果距离为0，那么权重就是1
                    self.weight_assistant[x_i, c_i + 2] = 1

    def fit(self, X):
        # 初始化模糊聚类的权重集
        if self.cluster_type == 'Fuzzy-K-Means':
            # 0 存放最小的类别，1 存放对应的距离
            self.weight_assistant = np.zeros((X.shape[0], self.center_points.shape[0] + 2), dtype=np.float64)
            # 初期化每个样本对于不同中心点的权重为1/c
            self.weight_assistant[:, 2 : self.center_points.shape[0] + 2] = 1 / self.center_points.shape[0]

        self.step = 1
        while True:
            self.category_data.clear()
            for i in range(len(self.center_points)):
                self.category_data.append([])

            # 保存原中心点
            old_weight_assistant = self.weight_assistant.copy()
            for i, x in enumerate(X):
                distance = []
                # 计算每个点和重点的距离，K表示类别
                for k, center in enumerate(self.center_points):
                    if self.cluster_type == 'Fuzzy-K-Means':
                        # 计算每个点与中心点带权重的距离
                        weight = np.power(self.weight_assistant[i, k + 2], self.b)
                        euclid_square = np.power(self.distance_function.calculate_distance(x, center), 2)
                        distance.append(weight * euclid_square)
                    else:
                        distance.append(self.distance_function.calculate_distance(x, center))

                # 把数据放到对应的类中
                k = np.argmin(distance)
                if len(self.y) < len(X):
                    self.y.append(k)
                else:
                    self.y[i] = k

                # 保存类别和距离
                if self.cluster_type == 'Fuzzy-K-Means':
                    self.weight_assistant[i, 0:2] = k, np.min(distance)
                self.category_data[k].append(x)

                # 更新每个点权重
                self.weight_update(X)

            # 保存原中心点
            old_center_points = self.center_points.copy()
            # 求新的中心点
            if self.cluster_type == 'Fuzzy-K-Means':
                c = np.zeros((X.shape[1] + 1, self.center_points.shape[0]))
                for i, x in enumerate(X):
                    x_c = int(self.weight_assistant[i, 0])
                    weight = np.power(self.weight_assistant[i, x_c + 2], self.b)
                    # 每个样本x_i的所属类别的权重和
                    c[-1, x_c] += weight
                    # 乘以权重的点
                    c[:len(x), x_c] += weight * x
                for k in range(self.center_points.shape[0]):
                    # 如果中心点的权重为0，则不更新
                    if c[-1, k] != 0:
                        # 计算平均权重
                        weight_mean = c[:, k] / c[-1, k]
                        # 更新中心点
                        self.center_points[k, :] = weight_mean[:-1]
            else:
                for i, data_list in enumerate(self.category_data):
                    data_list = np.array(data_list)
                    if len(data_list) > 0:
                        # 更新聚类中心点
                        self.center_points[i] = np.mean(data_list, axis=0)

            convergence = []

            if self.cluster_type == 'Fuzzy-K-Means':
                center_mean = np.mean(abs(old_center_points - center_point), axis=1)
                label_div = np.sum(abs(old_weight_assistant[:, 0] - self.weight_assistant[:, 0]))
                # label_div = 1
                # 中心点偏动小于阈值或标签不在变化则收敛
                if all(abs(center_mean) <= self.center_threshold) or label_div == 0:
                    self.y = np.array(self.y)
                    return self
            else:
                # 判断中心点是否发生变化
                for i, center in enumerate(zip(old_center_points, self.center_points)):
                    convergence.append(np.all(center[0] == center[1]))

                # 收敛或者迭代次数超过最大次数
                if np.all(convergence):
                    self.y = np.array(self.y)
                    return self

            # 迭代次数超过最大次数
            if self.step >= self.max_iter:
                self.y = np.array(self.y)
                return self

            self.step += 1


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    dataset = pd.read_csv('datasets/ex_5.csv')

    # 距离方法 Euclid Gaussian
    distance_type = 'Gaussian'
    # 0.001 0.01 0.1 1 10 100
    beta = 100
    # K-Means Fuzzy-K-Means
    cluster_type = 'Fuzzy-K-Means'
    # Fuzzy-K-Means的b值 0 2
    b = 2

    fig, ax_list = plt.subplots(2, 2, layout='constrained', subplot_kw=dict(projection="3d"))
    X = dataset.loc[:, 'x1':'x3'].to_numpy()
    center_points_list = [np.array([[1, 1, 1], [-1, 1, -1]], dtype=np.float64),
                          np.array([[0, 0, 0], [1, 1, -1]], dtype=np.float64),
                          np.array([[0, 0, 0], [1, 1, 1], [-1, 0, 2]], dtype=np.float64),
                          np.array([[-0.1, 0, 0.1], [0, -0.1, 0.1], [-0.1, -0.1, 0.1]], dtype=np.float64)]
    model_list = []
    marker_list = ['o', '^', '*']

    if distance_type == 'Euclid':
        for i, center_point in enumerate(center_points_list):
            model = KCluster(cluster_type=cluster_type, center_points=center_point,
                             distance_function=DistanceFunction(
                                 distance_type=distance_type), b=b).fit(X)
            model_list.append(model)
    else:
        for i, center_point in enumerate(center_points_list):
            model = KCluster(cluster_type=cluster_type, center_points=center_point,
                             distance_function=DistanceFunction(
                                 distance_type=distance_type, beta=beta), b=b).fit(X)
            model_list.append(model)

    for i, (center_point, model) in enumerate(zip(center_points_list, model_list)):
        ax = ax_list[int(i / 2), int(i % 2)]
        for c in range(len(center_point)):
            x = X[np.where(model.y == c)]
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker=marker_list[c],
                       label='类别-{}，n={}'.format(c + 1, len(x)))
            ax.scatter(center_point[c, 0], center_point[c, 1], center_point[c, 2],
                       s=50, edgecolor='k', marker=marker_list[c],
                       label='类别-{}的中心点'.format(c + 1))
        ax.set_xlabel('X1 Label')
        ax.set_ylabel('X2 Label')
        ax.set_zlabel('X3 Label')
        ax.legend()

        category_statistics = '，'.join(['类别{}={}'.format(i, Counter(model.y)[label])
                                        for i, label in enumerate(Counter(model.y))])
        if distance_type == 'Euclid':
            if cluster_type == 'K-Means':
                ax.set_title('{}距离下，第{}题，c={} {}经过{}次迭代后收敛'.format(distance_type, i + 1,
                                                                   len(center_point),
                                                                   cluster_type, model.step))
                print('{}距离下，第{}题，c={} {}经过{}次迭代后收敛，{}'.format(distance_type, i + 1,
                                                               len(center_point),
                                                               cluster_type, model.step,
                                                               category_statistics))
            else:
                ax.set_title('{}距离下，第{}题，c={} b={} {}经过{}次迭代后收敛'.format(distance_type, i + 1,
                                                                        len(center_point), b,
                                                                        cluster_type, model.step))
                print('{}距离下，第{}题，c={} b={} {}经过{}次迭代后收敛，{}'.format(distance_type, i + 1,
                                                                    len(center_point), b,
                                                                    cluster_type, model.step,
                                                                    category_statistics))
        else:
            if cluster_type == 'K-Means':
                ax.set_title('{}距离下，beta={}，第{}题，c={} {}经过{}次迭代后收敛'.format(distance_type, beta,
                                                                           i + 1, len(center_point),
                                                                           cluster_type, model.step))
                print('{}距离下，beta={}，第{}题，c={} {}经过{}次迭代后收敛，{}'.format(distance_type, beta, i + 1,
                                                                       len(center_point),
                                                                       cluster_type, model.step,
                                                                       category_statistics))
            else:
                ax.set_title('{}距离下，beta={}，第{}题，c={} b={} {}经过{}次迭代后收敛'.format(distance_type, beta,
                                                                                i + 1, len(center_point), b,
                                                                                cluster_type, model.step))
                print('{}距离下，beta={}，第{}题，c={} b={} {}经过{}次迭代后收敛，{}'.format(distance_type, beta, i + 1,
                                                                            len(center_point), b,
                                                                            cluster_type, model.step,
                                                                            category_statistics))

    plt.show()

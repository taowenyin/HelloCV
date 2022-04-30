import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from datetime import datetime
from time import strftime
from time import gmtime
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.manifold import LocallyLinearEmbedding


class NormalPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        # 降维所需的基向量
        self.base_vectors = None

    # 均值归一化
    def mean_normalization(self, X):
        for j in range(self.n):
            me = np.mean(X[:, j])
            X[:, j] = X[:, j] - me
        return X

    # r为降低到的维数
    def fit(self, X):
        self.m = X.shape[0]
        self.n = X.shape[1]
        # 均值归一化
        X = self.mean_normalization(X)
        Xt = X.T
        # 协方差矩阵
        c = (1 / self.m) * Xt.dot(X)
        # 求解协方差矩阵的特征向量和特征值
        eigenvalue, featurevector = np.linalg.eig(c)
        # 对特征值索引排序 从大到小
        aso = np.argsort(eigenvalue)
        indexs = aso[::-1]
        eigenvalue_sum = np.sum(eigenvalue)
        self.base_vectors = []
        for i in range(self.n_components):
            self.base_vectors.append(featurevector[:, indexs[i]])  # 取前r个特征值大的特征向量作为基向量
        self.base_vectors = np.array(self.base_vectors, dtype=np.float64)
        return

    def transform(self, X):
        # r*n的P乘以n*m的矩阵转置后为m*r的矩阵
        return self.base_vectors.dot(X.T).T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def build_dataset_1(test_size=0.2):
    X, y = [], []
    scaler = StandardScaler()

    # 读取每个图像的类别
    for c in range(1, 41):
        # 读取每个图像
        for i in range(1, 11):
            # 每个图像的路径
            file = 'datasets/ORL_Faces/s{}/{}.pgm'.format(c, i)
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

            # 数据归一化
            img = scaler.fit_transform(img)
            img = img.reshape(1, -1)

            # 打开图像
            X.append(img[0])
            # 添加标签
            y.append(c)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def build_dataset_2(enlarge_size=5):
    X_train, X_test, y_train, y_test = [], [], [], []
    scaler = StandardScaler()

    enlarge_img_list = []
    enlarge_y_list = []
    org_img_list = []

    # 读取每个图像的类别
    for c in range(1, 41):
        # 读取每个图像
        for i in range(1, 11):
            # 每个图像的路径
            file = 'datasets/ORL_Faces/s{}/{}.pgm'.format(c, i)
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

            if i <= enlarge_size:
                enlarge_img = cv2.flip(img, flipCode=1)

                # 添加增广后和对对应的原始图片，以及标签
                enlarge_img_list.append(enlarge_img)
                org_img_list.append(img)
                enlarge_y_list.append(c)

                enlarge_img = scaler.fit_transform(enlarge_img).reshape(1, -1)
                org_img = scaler.fit_transform(img).reshape(1, -1)

                # 打开图像
                X_train.append(enlarge_img[0])
                X_train.append(org_img[0])
                # 添加标签
                y_train.append(c)
                y_train.append(c)
            else:
                # 数据归一化
                test_img = scaler.fit_transform(img).reshape(1, -1)
                X_test.append(test_img[0])
                y_test.append(c)

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    return X_train, X_test, y_train, y_test, enlarge_img_list, org_img_list, enlarge_y_list


def program_1():
    # 获得训练数据和测试数据
    X_train, X_test, y_train, y_test = build_dataset_1()

    # SVD方法
    start_time_svd = datetime.now()
    pca_svd = PCA(n_components=20, svd_solver='full')
    X_train_svd = pca_svd.fit_transform(X_train)
    end_time_svd = datetime.now()
    diff_sec_svd = (end_time_svd - start_time_svd).seconds
    model_svd = SVC()
    model_svd.fit(X_train_svd, y_train)
    X_test_svd = pca_svd.transform(X_test)
    score_svd = model_svd.score(X_test_svd, y_test)

    print('==========SVD==========')
    print('测试集上的预测精度为:{}'.format(score_svd))
    print('SVD降维时间:{}'.format(strftime("%H:%M:%S", gmtime(diff_sec_svd))))

    # 传统PCA方法
    start_time_pca = datetime.now()
    pca_norm = NormalPCA(n_components=20)
    X_train_pca = pca_norm.fit_transform(X_train)
    end_time_pca = datetime.now()
    diff_sec_pca = (end_time_pca - start_time_pca).seconds
    model_pca = SVC()
    model_pca.fit(X_train_pca, y_train)
    X_test_pca = pca_norm.transform(X_test)
    score_pca = model_pca.score(X_test_pca, y_test)

    print('==========PCA==========')
    print('测试集上的预测精度为:{}'.format(score_pca))
    print('PCA降维时间:{}'.format(strftime("%H:%M:%S", gmtime(diff_sec_pca))))


def program_2():
    # 获得训练数据和测试数据
    X_train, X_test, y_train, y_test, enlarge_img_list, org_img_list, enlarge_y_list = build_dataset_2()

    search_parameter = np.arange(1, 10, 2)
    param_dict = {'n_neighbors': search_parameter}
    # 把数据设为10组，每次90%的训练集和10%的验证集
    estimator = GridSearchCV(estimator=KNeighborsClassifier(algorithm='kd_tree'),
                                  param_grid=param_dict, cv=10, return_train_score=True)
    estimator.fit(X_train, y_train)
    estimator.score(X_test, y_test)

    best_score = estimator.best_score_
    best_params = estimator.best_params_

    print('在训练集上K近邻的最佳参数K = {}'.format(best_params['n_neighbors']))
    print('在测试集上，最佳参数的分类精度 = {:.3f}，误差率 = {:.3f}'.format(best_score, 1 - best_score))

    rows = 4
    columns = 25
    fig, ax_list = plt.subplots(rows, columns)
    for i in range(columns):
        # 读取坐标轴
        ax_org_ax_1 = ax_list[0, i]
        ax_enlarge_ax_1 = ax_list[1, i]
        ax_org_ax_2 = ax_list[2, i]
        ax_enlarge_ax_2 = ax_list[3, i]


        # 读取图片
        org_img_1 = org_img_list[i]
        enlarge_img_1 = enlarge_img_list[i]
        y_1 = enlarge_y_list[i]
        org_img_2 = org_img_list[i + columns]
        enlarge_img_2 = enlarge_img_list[i + columns]
        y_2 = enlarge_y_list[i + columns]

        # 显示图片
        ax_org_ax_1.imshow(cv2.cvtColor(org_img_1, cv2.COLOR_BGR2RGB))
        ax_org_ax_1.set_axis_off()
        ax_org_ax_1.set_title(str(y_1))
        ax_enlarge_ax_1.imshow(cv2.cvtColor(enlarge_img_1, cv2.COLOR_BGR2RGB))
        ax_enlarge_ax_1.set_axis_off()
        ax_enlarge_ax_1.set_title(str(y_1))

        ax_org_ax_2.imshow(cv2.cvtColor(org_img_2, cv2.COLOR_BGR2RGB))
        ax_org_ax_2.set_axis_off()
        ax_org_ax_2.set_title(str(y_2))
        ax_enlarge_ax_2.imshow(cv2.cvtColor(enlarge_img_2, cv2.COLOR_BGR2RGB))
        ax_enlarge_ax_2.set_axis_off()
        ax_enlarge_ax_2.set_title(str(y_2))

    plt.suptitle('采用翻转方式对图像进行增广，单数行是原始图，双数行是翻转图')
    plt.show()


def program_3():
    # 获得训练数据和测试数据
    X_train, X_test, y_train, y_test, enlarge_img_list, org_img_list, enlarge_y_list = build_dataset_2()

    search_parameter = np.arange(1, 10, 2)
    param_dict = {'n_neighbors': search_parameter}

    pca = PCA(n_components=20, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # 把数据设为10组，每次90%的训练集和10%的验证集
    estimator_pca = GridSearchCV(estimator=KNeighborsClassifier(algorithm='kd_tree'),
                             param_grid=param_dict, cv=10, return_train_score=True)
    estimator_pca.fit(X_train_pca, y_train)
    estimator_pca.score(X_test_pca, y_test)
    best_score_pca = estimator_pca.best_score_
    best_params_pca = estimator_pca.best_params_
    print('采用PCA降维后，在训练集上K近邻的最佳参数K = {}'.format(best_params_pca['n_neighbors']))
    print('采用PCA降维后，在测试集上，最佳参数的分类精度 = {:.3f}，误差率 = {:.3f}'.format(best_score_pca,
                                                                  1 - best_score_pca))

    lle = LocallyLinearEmbedding(n_components=20)
    X_train_lle = lle.fit_transform(X_train)
    X_test_lle = lle.transform(X_test)
    # 把数据设为10组，每次90%的训练集和10%的验证集
    estimator_lle = GridSearchCV(estimator=KNeighborsClassifier(algorithm='kd_tree'),
                                 param_grid=param_dict, cv=10, return_train_score=True)
    estimator_lle.fit(X_train_lle, y_train)
    estimator_lle.score(X_test_lle, y_test)
    best_score_lle = estimator_lle.best_score_
    best_params_lle = estimator_lle.best_params_

    print('采用LLE降维后，在训练集上K近邻的最佳参数K = {}'.format(best_params_lle['n_neighbors']))
    print('采用LLE降维后，在测试集上，最佳参数的分类精度 = {:.3f}，误差率 = {:.3f}'.format(best_score_lle,
                                                                  1 - best_score_lle))

def program_4():
    # 获得训练数据和测试数据
    X_train, X_test, y_train, y_test = build_dataset_1()

    # Step1：用TrSet获得投影阵M，用其重建TeSet，计算重建误差ETE
    pca_1 = PCA(n_components=20, svd_solver='full')
    # 使用训练集计算投影矩阵M
    X_train_pca_1 = pca_1.fit_transform(X_train)
    # 再使用投影矩阵M来重建测试集
    X_test_pca_recons_1 = pca_1.inverse_transform(pca_1.transform(X_test))
    # PCA的重建误差
    ete_pca = np.linalg.norm(X_test - X_test_pca_recons_1, ord=2)
    print('PCA下，用TrSet获得投影阵M，用其重建TeSet的重建误差ETE = {:.3f}'.format(ete_pca))

    lle_1 = LocallyLinearEmbedding(n_components=20)
    lle_1.fit_transform(X_train)
    ete_lle = lle_1.reconstruction_error_
    print('LLE下，用TrSet获得投影阵M，用其重建TeSet的重建误差ETE = {:.3f}'.format(ete_lle))

    # Step2：用TrSet+TeSet获得投影阵M+，用其重建TeSet，计算重建误差ETE+
    X_train_plus = np.vstack((X_train, X_test))
    pca_2 = PCA(n_components=20, svd_solver='full')
    # 使用训练集计算投影矩阵M+
    X_train_pca_2 = pca_2.fit_transform(X_train_plus)
    # 再使用投影矩阵M来重建测试集
    X_test_pca_recons_2 = pca_2.inverse_transform(pca_2.transform(X_test))
    # PCA的重建误差
    ete_pca_plus = np.linalg.norm(X_test - X_test_pca_recons_2, ord=2)
    print('PCA下，用TrSet+TeSet获得投影阵M+，用其重建TeSet，计算重建误差ETE+ = {:.3f}'.format(ete_pca_plus))

    lle_2 = LocallyLinearEmbedding(n_components=20)
    lle_2.fit_transform(X_train_plus)
    ete_lle_plus = lle_1.reconstruction_error_
    print('LLE下，用TrSet+TeSet获得投影阵M+，用其重建TeSet，计算重建误差ETE+ = {:.3f}'.format(ete_lle_plus))


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 第一题
    # program_1()

    # 第二题
    # program_2()

    # 第三题
    # program_3()

    # 第三题
    program_4()



import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def build_test_dataset(test_size=10):
    # 获取测试集数据
    test_X_1 = np.random.uniform(0, 1, test_size)
    test_X_2 = np.random.uniform(0, 1, test_size)
    test_y_1 = np.ones(test_X_1.shape)
    test_y_2 = np.full(test_X_2.shape, -1)

    test_X = np.append(test_X_1, test_X_2).reshape(-1, 1)
    test_y = np.append(test_y_1, test_y_2).reshape(-1, 1)
    test_data = np.hstack((test_X, test_y))
    # 把数据打乱
    np.random.shuffle(test_data)

    X_test = np.array(test_data[:, 0]).reshape(-1, 1)
    y_test = np.array(test_data[:, 1], dtype=np.int64).reshape(-1, 1)

    return X_test, y_test


def build_train_dataset(train_size=100, gama=0.1):
    # 获取训练集和验证集数据
    train_X_1 = np.random.uniform(0, 1, int(train_size / 2))
    train_X_2 = np.random.uniform(0, 1, int(train_size / 2))
    train_y_1 = np.ones(train_X_1.shape)
    train_y_2 = np.full(train_X_2.shape, -1)

    X_all_train = np.append(train_X_1, train_X_2).reshape(-1, 1)
    y_all_train = np.append(train_y_1, train_y_2)

    X_train, X_val, y_train, y_val = train_test_split(X_all_train, y_all_train, test_size=gama)

    return X_all_train, y_all_train, X_train, X_val, y_train, y_val


if __name__ == '__main__':
    train_size, test_size = 100, 10

    # 获取数据集
    X_test, y_test = build_test_dataset(test_size)
    X_all_train, y_all_train, X_train, X_val, y_train, y_val = build_train_dataset(train_size)

    # 设置CV模型
    search_parameter = np.arange(1, int(train_size * 0.9), 2)
    param_dict = {'n_neighbors': search_parameter}
    # 把数据设为10组，每次90%的训练集和10%的验证集
    estimator_gscv = GridSearchCV(estimator=KNeighborsClassifier(algorithm='kd_tree'),
                                  param_grid=param_dict, cv=10, return_train_score=True)
    estimator_gscv.fit(X_all_train, y_all_train)
    estimator_gscv.score(X_test, y_test)

    best_score = estimator_gscv.best_score_
    best_params = estimator_gscv.best_params_
    cv_results = estimator_gscv.cv_results_

    print('在训练集上K近邻的最佳参数K = {}'.format(best_params['n_neighbors']))
    print('在测试集上，最佳参数的分类精度 = {:.3f}，误差率 = {:.3f}'.format(best_score, 1 - best_score))

    mean_train_score = cv_results['mean_train_score']
    worst_train_score = np.min(mean_train_score)
    worst_k_index = np.argmin(mean_train_score)
    worst_k = search_parameter[worst_k_index]

    print('在训练集上K近邻的最差参数K = {}'.format(worst_k))
    print('在测试集上，最差参数的分类精度 = {:.3f}，误差率 = {:.3f}'.format(worst_train_score,
                                                         1 - worst_train_score))

    # 重复5次
    for i in range(5):
        print('==========重复5次的第{}次=========='.format(i + 1))

        X_test, y_test = build_test_dataset(test_size)
        X_all_train, y_all_train, X_train, X_val, y_train, y_val = build_train_dataset(train_size)

        estimator_gscv.fit(X_all_train, y_all_train)
        estimator_gscv.score(X_test, y_test)

        best_score = estimator_gscv.best_score_
        best_params = estimator_gscv.best_params_
        cv_results = estimator_gscv.cv_results_

        print('在训练集上K近邻的最佳参数K = {}'.format(best_params['n_neighbors']))
        print('在测试集上，最佳参数的分类精度 = {:.3f}，误差率 = {:.3f}'.format(best_score,
                                                             1 - best_score))

        mean_train_score = cv_results['mean_train_score']
        worst_train_score = np.min(mean_train_score)
        worst_k_index = np.argmin(mean_train_score)
        worst_k = search_parameter[worst_k_index]

        print('在训练集上K近邻的最差参数K = {}'.format(worst_k))
        print('在测试集上，最差参数的分类精度 = {:.3f}，误差率 = {:.3f}'.format(worst_train_score,
                                                             1 - worst_train_score))

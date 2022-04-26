import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = 2

    a = np.dot(y, x)

    print('xxx')
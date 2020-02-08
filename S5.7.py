import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
重映射、仿射
'''
if __name__ == '__main__':
    plt_rows = 2
    plt_columns = 2

    lena = cv2.imread('data/Lena.png')
    plt.subplot(plt_rows, plt_columns, 1)
    plt.title('Lena-IMG')
    plt.imshow(cv2.cvtColor(lena.copy(), cv2.COLOR_BGR2RGB))

    # 获取图像的大小
    rows, cols, depth = lena.shape
    # 创建X、Y的映射
    mapX = np.zeros((rows, cols), np.float32)
    mapY = np.zeros((rows, cols), np.float32)

    # 创建映射表
    for i in range(rows):
        for j in range(cols):
            mapX.itemset((i, j), j)
            mapY.itemset((i, j), rows - i)
    # 重映射
    remap = cv2.remap(lena, mapX, mapY, cv2.INTER_LINEAR)
    plt.subplot(plt_rows, plt_columns, 2)
    plt.title('ReMap')
    plt.imshow(cv2.cvtColor(remap.copy(), cv2.COLOR_BGR2RGB))

    # 仿射矩阵
    M = np.array(
        [
            [0.5, 0, 0],
            [0, 1, 0]
        ],
        dtype='float32'
    )
    affine_1 = cv2.warpAffine(lena, M, (rows, cols))
    plt.subplot(plt_rows, plt_columns, 3)
    plt.title('Affine-1')
    plt.imshow(cv2.cvtColor(affine_1.copy(), cv2.COLOR_BGR2RGB))

    # 根据三点进行仿射
    srcPoint = np.array(
        [[0, 0], [cols, 0], [0, rows]],
        dtype='float32'
    )
    dstPoint = np.array(
        [[0, rows], [0, 0], [cols, rows]],
        dtype='float32'
    )
    # 根据三点计算得到仿射矩阵
    M = cv2.getAffineTransform(srcPoint, dstPoint)
    affine_2 = cv2.warpAffine(lena, M, (rows, cols))
    plt.subplot(plt_rows, plt_columns, 4)
    plt.title('Affine-2')
    plt.imshow(cv2.cvtColor(affine_2.copy(), cv2.COLOR_BGR2RGB))

    plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 腐烛与膨胀
if __name__ == '__main__':
    rows = 1
    columns = 3

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    # 创建一个卷积核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    print(kernel)

    '''
    腐烛
    ksize：巻积核大小
    iterations：迭代次数
    '''
    # 实现腐烛，iterations表示迭代次数
    erode = cv2.erode(lena, kernel, iterations=1)
    plt.subplot(rows, columns, 2)
    plt.title('Erode')
    plt.imshow(cv2.cvtColor(erode, cv2.COLOR_BGR2RGB))

    '''
    膨胀
    ksize：巻积核大小
    iterations：迭代次数
    '''
    # 实现腐烛，iterations表示迭代次数
    dilate = cv2.dilate(lena, kernel, iterations=1)
    plt.subplot(rows, columns, 3)
    plt.title('Dilate')
    plt.imshow(cv2.cvtColor(dilate, cv2.COLOR_BGR2RGB))

    plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 非线性滤波
if __name__ == '__main__':
    rows = 1
    columns = 3

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    '''
    中值滤波
    ksize：巻积核大小
    每个像素的灰度值进行大小排序，按照排序的中间值作为新值
    '''
    median = cv2.medianBlur(lena, 7)
    plt.subplot(rows, columns, 2)
    plt.title('Median')
    plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))

    '''
    双边滤波
    d：每个像素领域的直径，该值<0时，那么有sigmaSpace计算得来
    sigmaColor：空间高斯函数标准差，该值越大就表明邻域内越宽广的颜色会被计算
    sigmaSpace：灰度值相似性高斯函数标准差，该值越大就表明邻域颜色足够相近的的颜色的影响越大
    '''
    bilateral = cv2.bilateralFilter(lena, 25, 25 * 2, 25)
    plt.subplot(rows, columns, 3)
    plt.title('Bilateral')
    plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))

    plt.show()
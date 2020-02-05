import cv2
import numpy as np
import matplotlib.pyplot as plt


# 线性滤波 g = f x kernel
if __name__ == '__main__':
    rows = 2
    columns = 2

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    '''
    方框滤波
    ddepth：图像深度
    ksize：巻积核大小
    normalize：是否进行归一化
                 [1 1 ... 1
    kernel = a *  .........
                  1 1 ... 1]
    normalize = true, a = 1 / (ksize_width * ksize_height)
    normalize = false, a = 1
    '''
    box = cv2.boxFilter(lena, -1, (7, 7), normalize=True)
    plt.subplot(rows, columns, 2)
    plt.title('Box')
    plt.imshow(cv2.cvtColor(box, cv2.COLOR_BGR2RGB))

    '''
    均值滤波
    ddepth：图像深度
    ksize：巻积核大小
    normalize：是否进行归一化
                 [1 1 ... 1
    kernel = a *  .........
                  1 1 ... 1]
    a = 1 / (ksize_width * ksize_height)
    '''
    blur = cv2.blur(lena, (7, 7))
    plt.subplot(rows, columns, 3)
    plt.title('Blur')
    plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))

    '''
    高斯滤波
    ksize：巻积核大小
    sigmaX：高斯核在X方向的标准偏差，一般ksize / sigmaX = 3 左右
    '''
    gaussian = cv2.GaussianBlur(lena, (7, 7), 2)
    plt.subplot(rows, columns, 4)
    plt.title('Gaussian')
    plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))

    plt.show()
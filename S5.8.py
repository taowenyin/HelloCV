import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
直方图均衡化
'''
if __name__ == '__main__':
    plt_rows = 2
    plt_columns = 3

    lena = cv2.imread('data/Lena.png')
    plt.subplot(plt_rows, plt_columns, 1)
    plt.title('Lena-IMG')
    plt.imshow(cv2.cvtColor(lena.copy(), cv2.COLOR_BGR2RGB))

    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    plt.subplot(plt_rows, plt_columns, 2)
    plt.title('Lena-Gray')
    plt.imshow(cv2.cvtColor(lena_gray.copy(), cv2.COLOR_BGR2RGB))

    # 绘制图像的直方图
    plt.subplot(plt_rows, plt_columns, 3)
    plt.title('Lena-Hist')
    plt.hist(lena_gray.ravel(), 256, [0, 256])

    # 直方图均衡化
    equalize = cv2.equalizeHist(lena_gray)
    plt.subplot(plt_rows, plt_columns, 4)
    plt.title('Lena-Equ')
    plt.imshow(cv2.cvtColor(equalize.copy(), cv2.COLOR_BGR2RGB))

    # 绘制图像的直方图
    plt.subplot(plt_rows, plt_columns, 5)
    plt.title('Lena-Equ-Hist')
    plt.hist(equalize.ravel(), 256, [0, 256])

    plt.show()
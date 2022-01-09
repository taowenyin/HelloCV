import cv2
import numpy as np
import matplotlib.pyplot as plt


# 阈值化
if __name__ == '__main__':
    rows = 1
    columns = 3

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    # 转化为灰度图
    lenaGray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # 固定阈值
    threshold = cv2.threshold(lenaGray.copy(), 150, 255, cv2.THRESH_BINARY)
    plt.subplot(rows, columns, 2)
    plt.title('Threshold')
    plt.imshow(cv2.cvtColor(threshold[1], cv2.COLOR_BGR2RGB))

    # 自适应阈值
    adaptiveThreshold = cv2.adaptiveThreshold(lenaGray.copy(), 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.subplot(rows, columns, 3)
    plt.title('AdaptiveThreshold')
    plt.imshow(cv2.cvtColor(adaptiveThreshold, cv2.COLOR_BGR2RGB))

    plt.show()
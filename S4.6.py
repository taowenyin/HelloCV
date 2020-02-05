import cv2
import numpy as np
import matplotlib.pyplot as plt


# 图像金字塔
if __name__ == '__main__':
    rows = 1
    columns = 3

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    # 缩小图片
    low_resize_logo = cv2.resize(lena, (200, 200), interpolation=cv2.INTER_AREA)
    plt.subplot(rows, columns, 2)
    plt.title('Low Resize')
    plt.imshow(cv2.cvtColor(low_resize_logo, cv2.COLOR_BGR2RGB))

    # 放大图片
    up_resize_logo = cv2.resize(lena, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    plt.subplot(rows, columns, 3)
    plt.title('Up Resize')
    plt.imshow(cv2.cvtColor(up_resize_logo, cv2.COLOR_BGR2RGB))

    plt.show()
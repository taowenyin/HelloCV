import cv2
import numpy as np
import matplotlib.pyplot as plt


# 亮度、对比度调整 g(x) = contrast * f(x) + bright
if __name__ == '__main__':
    rows = 1
    columns = 3

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    # 对比度的值
    contrast = 50
    # 亮度值
    bright = 0

    # clip表示把数组数据进行拷贝，如果比min小，那么就是min值，如果比max大，那么就是max值
    contrast_lena = np.uint8(np.clip(((contrast * 0.01) * lena + bright), 0, 255))
    plt.subplot(rows, columns, 2)
    plt.title('Contrast')
    plt.imshow(cv2.cvtColor(contrast_lena, cv2.COLOR_BGR2RGB))

    # 对比度的值
    contrast = 100
    # 亮度值
    bright = 50

    bright_lena = np.uint8(np.clip(((contrast * 0.01) * lena + bright), 0, 255))
    plt.subplot(rows, columns, 3)
    plt.title('Bright')
    plt.imshow(cv2.cvtColor(bright_lena, cv2.COLOR_BGR2RGB))

    plt.show()
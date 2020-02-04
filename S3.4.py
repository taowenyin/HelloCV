import cv2
import numpy as np
import matplotlib.pyplot as plt


# 色彩的分离与合并
if __name__ == '__main__':
    rows = 1
    columns = 3

    lena = cv2.imread('data/Lena.png')

    # 创建图像大小的0空间
    zeros = np.zeros(lena.shape[:2], dtype='uint8')

    # 分离出图像的各通道数据
    B, G, R = cv2.split(lena)

    # 合并成完整的数据
    blue = cv2.merge([B, zeros, zeros])
    green = cv2.merge([zeros, G, zeros])
    red = cv2.merge([zeros, zeros, R])

    plt.subplot(rows, columns, 1)
    plt.title('Blue-Channel')
    plt.imshow(cv2.cvtColor(blue, cv2.COLOR_BGR2RGB))

    plt.subplot(rows, columns, 2)
    plt.title('Green-Channel')
    plt.imshow(cv2.cvtColor(green, cv2.COLOR_BGR2RGB))

    plt.subplot(rows, columns, 3)
    plt.title('Red-Channel')
    plt.imshow(cv2.cvtColor(red, cv2.COLOR_BGR2RGB))

    plt.show()
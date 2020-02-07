import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
Laplacian算子
当ksize = 1时
Laplacian采用3x3孔径
[0  1 0
 1 -4 1
 0  1 0]
'''
if __name__ == '__main__':
    rows = 2
    columns = 2

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena.copy(), cv2.COLOR_BGR2RGB))

    lena_blur = cv2.GaussianBlur(lena, (3, 3), 2)
    plt.subplot(rows, columns, 2)
    plt.title('Lena-Blur')
    plt.imshow(cv2.cvtColor(lena_blur.copy(), cv2.COLOR_BGR2RGB))

    lena_gray = cv2.cvtColor(lena_blur, cv2.COLOR_BGR2GRAY)
    plt.subplot(rows, columns, 3)
    plt.title('Lena-Gray')
    plt.imshow(cv2.cvtColor(lena_gray.copy(), cv2.COLOR_BGR2RGB))

    laplacian = cv2.Laplacian(lena_gray, cv2.CV_32F)
    # 计算绝对值
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    plt.subplot(rows, columns, 4)
    plt.title('Laplacian')
    plt.imshow(cv2.cvtColor(laplacian_abs.copy(), cv2.COLOR_BGR2RGB))

    plt.show()
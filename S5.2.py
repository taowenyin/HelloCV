import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
Sobel算子
灰度值的内积计算
     [-1 0 +1
Gx =  -2 0 +2  * I
      -1 0 +1]
      
     [+1 +2 +1
Gy =   0  0  0  * I
      -1 -2 -1]

G = |Gx| + |Gy|
'''
if __name__ == '__main__':
    rows = 1
    columns = 4

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena.copy(), cv2.COLOR_BGR2RGB))

    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    plt.subplot(rows, columns, 2)
    plt.title('Lena-Gray')
    plt.imshow(cv2.cvtColor(lena_gray.copy(), cv2.COLOR_BGR2RGB))

    # 求x方向的梯度
    sobelX = cv2.Sobel(lena_gray, cv2.CV_32F, 1, 0, ksize=3)
    plt.subplot(rows, columns, 3)
    plt.title('Sobel-X')
    plt.imshow(cv2.cvtColor(sobelX.copy(), cv2.COLOR_BGR2RGB))

    # 求y方向的梯度
    sobelY = cv2.Sobel(lena_gray, cv2.CV_32F, 0, 1, ksize=3)
    plt.subplot(rows, columns, 4)
    plt.title('Sobel-Y')
    plt.imshow(cv2.cvtColor(sobelY.copy(), cv2.COLOR_BGR2RGB))

    plt.show()
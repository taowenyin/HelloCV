import numpy as np
import cv2
import matplotlib.pyplot as plt


# 平滑处理
if __name__ == "__main__":
    rows = 2
    columns = 4

    lena = cv2.imread("data/Lena.png")
    plt.subplot(rows, columns, 1)
    plt.title("Lena")
    plt.imshow(cv2.cvtColor(lena.copy(), cv2.COLOR_BGR2RGB))

    # 均值滤波
    lena = cv2.imread("data/Lena.png")
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.subplot(rows, columns, 2)
    plt.title("Lena-Blur")
    plt.imshow(cv2.blur(lena, ksize=(3, 3)))

    # 方框滤波
    lena = cv2.imread("data/Lena.png")
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.subplot(rows, columns, 3)
    plt.title("Lena-BoxFilter")
    plt.imshow(cv2.boxFilter(lena, ddepth=-1, ksize=(5, 5), normalize=True))

    # 高斯滤波
    lena = cv2.imread("data/Lena.png")
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.subplot(rows, columns, 4)
    plt.title("Lena-Gaussian")
    plt.imshow(cv2.GaussianBlur(lena, ksize=(5, 5), sigmaX=0, sigmaY=0))

    # 中值滤波
    lena = cv2.imread("data/Lena.png")
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.subplot(rows, columns, 5)
    plt.title("Lena-Median")
    plt.imshow(cv2.medianBlur(lena, ksize=5))

    # 双边滤波
    lena = cv2.imread("data/Lena.png")
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.subplot(rows, columns, 6)
    plt.title("Lena-Median")
    plt.imshow(cv2.bilateralFilter(lena, d=5, sigmaColor=100, sigmaSpace=100))

    # 2D卷积滤波
    lena = cv2.imread("data/Lena.png")
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.subplot(rows, columns, 7)
    plt.title("Lena-Conv")
    kernel = np.ones((9, 9), np.float32) / 81
    plt.imshow(cv2.filter2D(lena, -1, kernel=kernel))

    plt.show()

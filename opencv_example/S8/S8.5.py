import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    rows = 1
    columns = 2

    src = cv2.imread("data/gradient.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org1")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    # 梯度运算
    src = cv2.imread("data/gradient.png")
    plt.subplot(rows, columns, 2)
    plt.title("Gradient")
    kernel = np.ones((10, 10), np.uint8)
    plt.imshow(cv2.morphologyEx(src=src, op=cv2.MORPH_GRADIENT, kernel=kernel))

    plt.show()

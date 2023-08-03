import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    rows = 1
    columns = 4

    src = cv2.imread("data/opening2.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    # 自定义核
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (60, 60))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))

    src = cv2.imread("data/opening2.png")
    plt.subplot(rows, columns, 2)
    plt.title("Kernel-Rect")
    kernel = np.ones((10, 10), np.uint8)
    plt.imshow(cv2.morphologyEx(src=src, op=cv2.MORPH_DILATE, kernel=kernel1))

    src = cv2.imread("data/opening2.png")
    plt.subplot(rows, columns, 3)
    plt.title("Kernel-Cross")
    kernel = np.ones((5, 5), np.uint8)
    plt.imshow(cv2.morphologyEx(src=src, op=cv2.MORPH_DILATE, kernel=kernel2))

    src = cv2.imread("data/opening2.png")
    plt.subplot(rows, columns, 4)
    plt.title("Kernel-Ellipse")
    kernel = np.ones((5, 5), np.uint8)
    plt.imshow(cv2.morphologyEx(src=src, op=cv2.MORPH_DILATE, kernel=kernel3))

    plt.show()

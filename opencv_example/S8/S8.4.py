import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    rows = 2
    columns = 2

    src = cv2.imread("data/closing1.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org1")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/closing2.png")
    plt.subplot(rows, columns, 2)
    plt.title("Org2")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    # 闭运算
    src = cv2.imread("data/closing1.png")
    plt.subplot(rows, columns, 3)
    plt.title("Closing1")
    kernel = np.ones((10, 10), np.uint8)
    plt.imshow(cv2.morphologyEx(src=src, op=cv2.MORPH_CLOSE, kernel=kernel))

    # 闭运算
    src = cv2.imread("data/closing2.png")
    plt.subplot(rows, columns, 4)
    plt.title("Closing2")
    kernel = np.ones((10, 10), np.uint8)
    plt.imshow(
        cv2.morphologyEx(src=src, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=3)
    )

    plt.show()

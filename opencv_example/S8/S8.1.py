import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    rows = 1
    columns = 2

    src = cv2.imread("data/erode.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    # 腐蚀
    src = cv2.imread("data/erode.png")
    plt.subplot(rows, columns, 2)
    plt.title("Erode")
    kernel = np.ones((5, 5), np.uint8)
    plt.imshow(cv2.erode(src=src, kernel=kernel))

    plt.show()

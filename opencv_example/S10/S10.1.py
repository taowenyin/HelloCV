import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rows = 1
    columns = 2

    src = cv2.imread("data/Lena.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY))

    src = cv2.imread("data/Lena.png", cv2.IMREAD_GRAYSCALE)
    plt.subplot(rows, columns, 2)
    plt.title("Canny")
    plt.imshow(cv2.Canny(src, 100, 200))

    plt.show()

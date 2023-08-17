import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rows = 2
    columns = 2

    src = cv2.imread("data/equ.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/equ.png", cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(src)
    plt.subplot(rows, columns, 2)
    plt.title("Equ")
    plt.imshow(cv2.cvtColor(equ.copy(), cv2.COLOR_BGR2RGB))

    plt.subplot(rows, columns, 3)
    plt.title("Src Hist")
    plt.hist(src.ravel(), 256, (0, 255))

    plt.subplot(rows, columns, 4)
    plt.title("Equ Hist")
    plt.hist(equ.ravel(), 256, (0, 255))

    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rows = 2
    columns = 4

    src = cv2.imread("data/Lena.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/Lena.png")
    r1 = cv2.pyrDown(src)
    plt.subplot(rows, columns, 2)
    plt.title("pyrDown 1")
    plt.imshow(cv2.cvtColor(r1, cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/Lena.png")
    r1 = cv2.pyrDown(src)
    r2 = cv2.pyrDown(r1)
    plt.subplot(rows, columns, 3)
    plt.title("pyrDown 2")
    plt.imshow(cv2.cvtColor(r2, cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/Lena.png")
    r1 = cv2.pyrDown(src)
    r2 = cv2.pyrDown(r1)
    r3 = cv2.pyrDown(r2)
    plt.subplot(rows, columns, 4)
    plt.title("pyrDown 3")
    plt.imshow(cv2.cvtColor(r3, cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/Lena.png")
    plt.subplot(rows, columns, 5)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/Lena.png")
    r1 = cv2.pyrUp(src)
    plt.subplot(rows, columns, 6)
    plt.title("pyrUp 1")
    plt.imshow(cv2.cvtColor(r1, cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/Lena.png")
    r1 = cv2.pyrUp(src)
    r2 = cv2.pyrUp(r1)
    plt.subplot(rows, columns, 7)
    plt.title("pyrUp 2")
    plt.imshow(cv2.cvtColor(r2, cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/Lena.png")
    r1 = cv2.pyrUp(src)
    r2 = cv2.pyrUp(r1)
    r3 = cv2.pyrUp(r2)
    plt.subplot(rows, columns, 8)
    plt.title("pyrUp 3")
    plt.imshow(cv2.cvtColor(r3, cv2.COLOR_BGR2RGB))

    plt.show()

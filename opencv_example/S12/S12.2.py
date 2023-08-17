import cv2
import numpy as np
import matplotlib.pyplot as plt


# 提取前景图像
if __name__ == "__main__":
    rows = 1
    columns = 3

    src = cv2.imread("data/loc1.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/loc1.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros(src.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    plt.subplot(rows, columns, 2)
    plt.title("Mask")
    plt.imshow(mask)

    src = cv2.imread("data/loc1.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros(src.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    loc = cv2.bitwise_and(src, mask)
    plt.subplot(rows, columns, 3)
    plt.title("Loc")
    plt.imshow(cv2.cvtColor(loc, cv2.COLOR_BGR2RGB))

    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 计算轮廓的长度
if __name__ == "__main__":
    rows = 1
    columns = 4

    src = cv2.imread("data/moments.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/moments.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    for i in range(len(contours)):
        canvas = np.zeros(src.shape, np.uint8)
        contour_img = cv2.drawContours(canvas, contours, i, (255, 255, 255), 5)

        plt.subplot(rows, columns, 2 + i)
        plt.title("Contours {} length = {}".format(i, cv2.arcLength(contours[i], True)))
        plt.imshow(contour_img)

    plt.show()

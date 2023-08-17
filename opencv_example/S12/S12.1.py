import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rows = 1
    columns = 5

    src = cv2.imread("data/contours.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/contours.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    plt.subplot(rows, columns, 2)
    plt.title("Draw Contours")
    plt.imshow(cv2.drawContours(src, contours, -1, (0, 0, 255), 5))

    src = cv2.imread("data/contours.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for i in range(len(contours)):
        # 创建一个与输入图像大小相同的空白画布
        canvas = np.zeros(src.shape, np.uint8)
        contour_img = cv2.drawContours(canvas, contours, i, (255, 255, 255), 5)

        plt.subplot(rows, columns, 3 + i)
        plt.title("Draw Contours {}".format(i))
        plt.imshow(contour_img)

    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rows = 1
    columns = 3

    src = cv2.imread("data/Lena.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY))

    src = cv2.imread("data/Lena.png", cv2.IMREAD_GRAYSCALE)
    plt.subplot(rows, columns, 2)
    plt.title("Scharr-XY") 
    dst_x = cv2.Scharr(src=src, ddepth=cv2.CV_64F, dx=1, dy=0)
    dst_y = cv2.Scharr(src=src, ddepth=cv2.CV_64F, dx=0, dy=1)
    plt.imshow(cv2.addWeighted(dst_x, 0.5, dst_y, 0.5, 0))

    src = cv2.imread("data/Lena.png", cv2.IMREAD_GRAYSCALE)
    plt.subplot(rows, columns, 3)
    plt.title("Scharr-XY-Abs")
    dst_x = cv2.Scharr(src=src, ddepth=cv2.CV_64F, dx=1, dy=0)
    dst_x = cv2.convertScaleAbs(dst_x)
    dst_y = cv2.Scharr(src=src, ddepth=cv2.CV_64F, dx=0, dy=1)
    dst_y = cv2.convertScaleAbs(dst_y)
    plt.imshow(cv2.addWeighted(dst_x, 0.5, dst_y, 0.5, 0))

    plt.show()

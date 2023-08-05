import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rows = 2
    columns = 4

    src = cv2.imread("data/sobel.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/sobel.png")
    plt.subplot(rows, columns, 2)
    plt.title("Sobel-X")
    plt.imshow(cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=1, dy=0))

    src = cv2.imread("data/sobel.png")
    plt.subplot(rows, columns, 3)
    plt.title("Sobel-X-Abs")
    plt.imshow(cv2.convertScaleAbs(cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=1, dy=0)))

    src = cv2.imread("data/sobel.png")
    plt.subplot(rows, columns, 4)
    plt.title("Sobel-Y")
    plt.imshow(cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=0, dy=1))

    src = cv2.imread("data/sobel.png")
    plt.subplot(rows, columns, 5)
    plt.title("Sobel-Y-Abs")
    plt.imshow(cv2.convertScaleAbs(cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=0, dy=1)))

    src = cv2.imread("data/sobel.png")
    plt.subplot(rows, columns, 6)
    plt.title("Sobel-XY")
    dst_x = cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=1, dy=0)
    dst_y = cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=0, dy=1)
    plt.imshow(cv2.addWeighted(dst_x, 0.5, dst_y, 0.5, 0))

    src = cv2.imread("data/sobel.png")
    plt.subplot(rows, columns, 7)
    plt.title("Sobel-XY-Abs")
    dst_x = cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=1, dy=0)
    dst_x = cv2.convertScaleAbs(dst_x)
    dst_y = cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=0, dy=1)
    dst_y = cv2.convertScaleAbs(dst_y)
    plt.imshow(cv2.addWeighted(dst_x, 0.5, dst_y, 0.5, 0))

    src = cv2.imread("data/Lena.png")
    plt.subplot(rows, columns, 8)
    plt.title("Lena")
    dst_x = cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=1, dy=0)
    dst_x = cv2.convertScaleAbs(dst_x)
    dst_y = cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=0, dy=1)
    dst_y = cv2.convertScaleAbs(dst_y)
    plt.imshow(cv2.addWeighted(dst_x, 0.5, dst_y, 0.5, 0))

    plt.show()

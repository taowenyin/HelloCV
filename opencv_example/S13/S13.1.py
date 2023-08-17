import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rows = 2
    columns = 2

    norain = cv2.imread("data/norain-1.png")
    plt.subplot(rows, columns, 1)
    plt.title("NoRain")
    plt.imshow(cv2.cvtColor(norain.copy(), cv2.COLOR_BGR2RGB))

    rain = cv2.imread("data/norain-1x2.png")
    plt.subplot(rows, columns, 2)
    plt.title("Rain")
    plt.imshow(cv2.cvtColor(rain.copy(), cv2.COLOR_BGR2RGB))

    plt.subplot(rows, columns, 3)
    plt.title("NoRain Hist")
    plt.hist(norain.ravel(), 256)

    plt.subplot(rows, columns, 4)
    plt.title("Rain Hist")
    plt.hist(rain.ravel(), 256)

    plt.show()

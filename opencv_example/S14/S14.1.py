import cv2
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft2, fftshift


if __name__ == "__main__":
    rows = 2
    columns = 4

    norain = cv2.imread("data/norain-1.png")
    plt.subplot(rows, columns, 1)
    plt.title("No Rain")
    plt.imshow(cv2.cvtColor(norain.copy(), cv2.COLOR_BGR2RGB))

    norain = cv2.imread("data/norain-1.png")
    plt.subplot(rows, columns, 2)
    plt.title("No Rain Gray")
    plt.imshow(cv2.cvtColor(norain.copy(), cv2.COLOR_BGR2GRAY))

    norain = cv2.imread("data/norain-1.png")
    norain = cv2.cvtColor(norain, cv2.COLOR_BGR2GRAY)
    norain_ff = fft2(norain)
    norain_ff = 20 * np.log(np.abs(norain_ff))
    plt.subplot(rows, columns, 3)
    plt.title("No Rain FF")
    plt.imshow(norain_ff)

    norain = cv2.imread("data/norain-1.png")
    norain = cv2.cvtColor(norain, cv2.COLOR_BGR2GRAY)
    norain_ff = fft2(norain)
    norain_ff = fftshift(norain_ff)
    norain_ff = 20 * np.log(np.abs(norain_ff))
    plt.subplot(rows, columns, 4)
    plt.title("No Rain FF Shitf")
    plt.imshow(norain_ff)

    rain = cv2.imread("data/norain-1x2.png")
    plt.subplot(rows, columns, 5)
    plt.title("Rain")
    plt.imshow(cv2.cvtColor(rain.copy(), cv2.COLOR_BGR2RGB))

    rain = cv2.imread("data/norain-1x2.png")
    plt.subplot(rows, columns, 6)
    plt.title("Rain Gray")
    plt.imshow(cv2.cvtColor(rain.copy(), cv2.COLOR_BGR2GRAY))

    rain = cv2.imread("data/norain-1x2.png")
    rain = cv2.cvtColor(rain, cv2.COLOR_BGR2GRAY)
    rain_ff = fft2(rain)
    rain_ff = 20 * np.log(np.abs(rain_ff))
    plt.subplot(rows, columns, 7)
    plt.title("Rain FF")
    plt.imshow(rain_ff)

    rain = cv2.imread("data/norain-1x2.png")
    rain = cv2.cvtColor(rain, cv2.COLOR_BGR2GRAY)
    rain_ff = fft2(rain)
    rain_ff = fftshift(rain_ff)
    rain_ff = 20 * np.log(np.abs(rain_ff))
    plt.subplot(rows, columns, 8)
    plt.title("Rain FF Shitf")
    plt.imshow(rain_ff)

    plt.show()

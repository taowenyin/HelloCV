import cv2
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft2, fftshift


if __name__ == "__main__":
    rows = 2
    columns = 4
    crop_size = 50

    norain = cv2.imread("data/norain-1.png")
    plt.subplot(rows, columns, 1)
    plt.title("NoRain")
    plt.imshow(cv2.cvtColor(norain.copy(), cv2.COLOR_BGR2RGB))

    norain = cv2.imread("data/norain-1.png")
    norain = cv2.cvtColor(norain, cv2.COLOR_BGR2GRAY)
    norain_ff = fft2(norain)
    norain_ff = fftshift(norain_ff)
    norain_ff = 20 * np.log(np.abs(norain_ff))
    plt.subplot(rows, columns, 2)
    plt.title("NoRain FF Shitf")
    plt.imshow(norain_ff)

    norain = cv2.imread("data/norain-1.png")
    norain = cv2.cvtColor(norain, cv2.COLOR_BGR2GRAY)
    norain_ff = fft2(norain)
    norain_ff = fftshift(norain_ff)
    H, W = norain.shape
    C_H, C_W = int(H / 2), int(W / 2)
    norain_ff[C_H - crop_size : C_H + crop_size, C_W - crop_size : C_W + crop_size] = 0
    norain_ff = 20 * np.log(np.abs(norain_ff))
    plt.subplot(rows, columns, 3)
    plt.title("Crop NoRain FF Shitf")
    plt.imshow(norain_ff)

    norain = cv2.imread("data/norain-1.png")
    norain = cv2.cvtColor(norain, cv2.COLOR_BGR2GRAY)
    norain_ff = fft2(norain)
    norain_ff = fftshift(norain_ff)
    H, W = norain.shape
    C_H, C_W = int(H / 2), int(W / 2)
    norain_ff[C_H - crop_size : C_H + crop_size, C_W - crop_size : C_W + crop_size] = 0
    norain_ff = np.fft.ifftshift(norain_ff)
    norain_ff = np.fft.ifft2(norain_ff)
    norain_ff = np.abs(norain_ff)
    plt.subplot(rows, columns, 4)
    plt.title("Reco NoRain FF Shitf")
    plt.imshow(norain_ff)

    rain = cv2.imread("data/norain-1x2.png")
    plt.subplot(rows, columns, 5)
    plt.title("Rain")
    plt.imshow(cv2.cvtColor(rain.copy(), cv2.COLOR_BGR2RGB))

    rain = cv2.imread("data/norain-1x2.png")
    rain = cv2.cvtColor(rain, cv2.COLOR_BGR2GRAY)
    rain_ff = fft2(rain)
    rain_ff = fftshift(rain_ff)
    rain_ff = 20 * np.log(np.abs(rain_ff))
    plt.subplot(rows, columns, 6)
    plt.title("Rain FF Shitf")
    plt.imshow(rain_ff)

    rain = cv2.imread("data/norain-1x2.png")
    rain = cv2.cvtColor(rain, cv2.COLOR_BGR2GRAY)
    rain_ff = fft2(rain)
    rain_ff = fftshift(rain_ff)
    H, W = rain.shape
    C_H, C_W = int(H / 2), int(W / 2)
    rain_ff[C_H - crop_size : C_H + crop_size, C_W - crop_size : C_W + crop_size] = 0
    rain_ff = 20 * np.log(np.abs(rain_ff))
    plt.subplot(rows, columns, 7)
    plt.title("Crop Rain FF Shitf")
    plt.imshow(rain_ff)

    rain = cv2.imread("data/norain-1x2.png")
    rain = cv2.cvtColor(rain, cv2.COLOR_BGR2GRAY)
    rain_ff = fft2(rain)
    rain_ff = fftshift(rain_ff)
    H, W = rain.shape
    C_H, C_W = int(H / 2), int(W / 2)
    rain_ff[C_H - crop_size : C_H + crop_size, C_W - crop_size : C_W + crop_size] = 0
    rain_ff = np.fft.ifftshift(rain_ff)
    rain_ff = np.fft.ifft2(rain_ff)
    rain_ff = np.abs(rain_ff)
    plt.subplot(rows, columns, 8)
    plt.title("Reco Rain FF Shitf")
    plt.imshow(rain_ff)

    plt.show()

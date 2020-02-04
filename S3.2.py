import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 读取图片
    lena = cv2.imread('data/Lena.png')
    plt.subplot(2, 5, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    # 读取Logo图片
    logo = cv2.imread('data/logo.png')
    plt.subplot(2, 5, 2)
    plt.title('Logo')
    plt.imshow(cv2.cvtColor(logo, cv2.COLOR_BGR2RGB))

    # 定义兴趣区域
    imgROI = lena[100: 100 + logo.shape[0], 100: 100 + logo.shape[1]]
    plt.subplot(2, 5, 3)
    plt.title('ROI')
    plt.imshow(cv2.cvtColor(imgROI, cv2.COLOR_BGR2RGB))

    # 图片变成灰度
    logoGray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 5, 4)
    plt.title('Logo Gray')
    plt.imshow(cv2.cvtColor(logoGray, cv2.COLOR_BGR2RGB))

    # 二值化Logo图片
    ret, mask = cv2.threshold(logoGray, 175, 255, cv2.THRESH_BINARY)
    plt.subplot(2, 5, 5)
    plt.title('Logo Mask')
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

    # 颜色取反
    mask_inv = cv2.bitwise_not(mask)
    plt.subplot(2, 5, 6)
    plt.title('Logo Mask Inv')
    plt.imshow(cv2.cvtColor(mask_inv, cv2.COLOR_BGR2RGB))

    # Logo的背景
    lena_bg = cv2.bitwise_and(imgROI, imgROI, mask=mask)
    plt.subplot(2, 5, 7)
    plt.title('Lena Background')
    plt.imshow(cv2.cvtColor(lena_bg, cv2.COLOR_BGR2RGB))

    # Logo的前景
    logo_fg = cv2.bitwise_and(logo, logo, mask=mask_inv)
    plt.subplot(2, 5, 8)
    plt.title('Lena Foreground')
    plt.imshow(cv2.cvtColor(logo_fg, cv2.COLOR_BGR2RGB))

    # 合并后的局部
    dest = cv2.add(lena_bg, logo_fg)
    plt.subplot(2, 5, 9)
    plt.title('Dest')
    plt.imshow(cv2.cvtColor(dest, cv2.COLOR_BGR2RGB))

    # 填回原图
    lena[100: 100 + logo.shape[0], 100: 100 + logo.shape[1]] = dest
    plt.subplot(2, 5, 10)
    plt.title('Final')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    plt.show()

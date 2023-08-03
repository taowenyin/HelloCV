import cv2
import numpy as np
import matplotlib.pyplot as plt


# 图片线性混合 g(x) = (1 - a)fa(x) + afb(x)
if __name__ == '__main__':
    rows = 3
    columns = 4

    # 读取图片
    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    # 读取Logo图片
    logo = cv2.imread('data/logo.png')
    plt.subplot(rows, columns, 2)
    plt.title('Logo')
    plt.imshow(cv2.cvtColor(logo, cv2.COLOR_BGR2RGB))

    '''
    普通线性组合
    '''
    # 缩放图片，线性组合要求图片大小相同
    resize_logo = cv2.resize(logo, (lena.shape[0], lena.shape[1]), interpolation=cv2.INTER_AREA)
    plt.subplot(rows, columns, 3)
    plt.title('Resize_Logo')
    plt.imshow(cv2.cvtColor(resize_logo, cv2.COLOR_BGR2RGB))

    # 获取线性组合结果
    resize_alpha = 0.8
    resize_beta = 1 - resize_alpha
    resize_gamma = 0
    dest = cv2.addWeighted(lena, resize_alpha, resize_logo, resize_beta, resize_gamma)
    plt.subplot(rows, columns, 4)
    plt.title('Rszie Dest')
    plt.imshow(cv2.cvtColor(dest, cv2.COLOR_BGR2RGB))

    '''
    ROI的线性组合
    '''
    # 定义兴趣区域
    imgROI = lena[100: 100 + logo.shape[0], 100: 100 + logo.shape[1]]
    plt.subplot(rows, columns, 5)
    plt.title('ROI')
    plt.imshow(cv2.cvtColor(imgROI, cv2.COLOR_BGR2RGB))

    # 图片变成灰度
    logoGray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    plt.subplot(rows, columns, 6)
    plt.title('Logo Gray')
    plt.imshow(cv2.cvtColor(logoGray, cv2.COLOR_BGR2RGB))

    # 二值化Logo图片
    ret, mask = cv2.threshold(logoGray, 175, 255, cv2.THRESH_BINARY)
    plt.subplot(rows, columns, 7)
    plt.title('Logo Mask')
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

    # 颜色取反
    mask_inv = cv2.bitwise_not(mask)
    plt.subplot(rows, columns, 8)
    plt.title('Logo Mask Inv')
    plt.imshow(cv2.cvtColor(mask_inv, cv2.COLOR_BGR2RGB))

    # Logo的背景
    lena_bg = cv2.bitwise_and(imgROI, imgROI, mask=mask)
    plt.subplot(rows, columns, 9)
    plt.title('Lena Background')
    plt.imshow(cv2.cvtColor(lena_bg, cv2.COLOR_BGR2RGB))

    # Logo的前景
    logo_fg = cv2.bitwise_and(logo, logo, mask=mask_inv)
    plt.subplot(rows, columns, 10)
    plt.title('Lena Foreground')
    plt.imshow(cv2.cvtColor(logo_fg, cv2.COLOR_BGR2RGB))

    # 合并后的局部
    roi_alpha = 0.4
    roi_beta = 1 - roi_alpha
    roi_gamma = 0
    dest = cv2.addWeighted(lena_bg, roi_alpha, logo_fg, roi_beta, roi_gamma)
    plt.subplot(rows, columns, 11)
    plt.title('ROI Dest')
    plt.imshow(cv2.cvtColor(dest, cv2.COLOR_BGR2RGB))

    # 填回原图
    lena[100: 100 + logo.shape[0], 100: 100 + logo.shape[1]] = dest
    plt.subplot(rows, columns, 12)
    plt.title('Final')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    plt.show()

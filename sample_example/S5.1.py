import cv2
import numpy as np
import matplotlib.pyplot as plt


# 边缘检测
if __name__ == '__main__':
    rows = 2
    columns = 3

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena.copy(), cv2.COLOR_BGR2RGB))

    # 第一步：把图像转化为灰度图像
    lean_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    plt.subplot(rows, columns, 2)
    plt.title('Lena-Gray')
    plt.imshow(cv2.cvtColor(lean_gray.copy(), cv2.COLOR_BGR2RGB))

    # 第二步：对图像进行降噪
    lean_blur = cv2.blur(lean_gray, (3, 3))
    plt.subplot(rows, columns, 3)
    plt.title('Lena-Blur')
    plt.imshow(cv2.cvtColor(lean_blur.copy(), cv2.COLOR_BGR2RGB))

    # 第三步：通过设置梯度大小和滞后阈值进行边缘检测
    lean_canny = cv2.Canny(lean_blur, 30, 80, apertureSize=3)
    plt.subplot(rows, columns, 4)
    plt.title('Lena-Canny')
    plt.imshow(cv2.cvtColor(lean_canny.copy(), cv2.COLOR_BGR2RGB))

    # 第四步：掩模取反
    canny_mask = cv2.bitwise_not(lean_canny.copy())
    plt.subplot(rows, columns, 5)
    plt.title('Lena-Mask')
    plt.imshow(cv2.cvtColor(canny_mask.copy(), cv2.COLOR_BGR2RGB))

    # 第五步：获取边框
    dst = cv2.copyTo(lena, canny_mask)
    plt.subplot(rows, columns, 6)
    plt.title('Lena-Dst')
    plt.imshow(cv2.cvtColor(dst.copy(), cv2.COLOR_BGR2RGB))

    plt.show()
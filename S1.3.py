import cv2


# 模糊图像
if __name__ == '__main__':
    # 读取图片
    lena = cv2.imread('data/Lena.png')

    # 实现模糊，Blur采用均值滤波，(7, 7)为均值滤波器的大小
    dest = cv2.blur(lena, (7, 7))

    # 现实图片
    cv2.imshow('Blur', dest)
    cv2.imshow('Normal', lena)
    cv2.waitKey(0)
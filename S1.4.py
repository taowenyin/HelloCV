import cv2


# 边缘检测
if __name__ == '__main__':
    # 读取图片
    lena = cv2.imread('data/Lena.png')

    # 转化为灰度图像
    gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    # 使用模糊进行降噪
    edge = cv2.blur(gray, (3, 3))

    canny = cv2.Canny(edge, 5, 80)

    # 现实图片
    cv2.imshow('Canny', canny)
    cv2.imshow('Normal', lena)
    cv2.waitKey(0)
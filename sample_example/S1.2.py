import cv2


# 腐烛图像
if __name__ == '__main__':
    # 读取图片
    lena = cv2.imread('data/Lena.png')

    # 创建一个卷积
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    print(kernel)

    # 实现腐烛，iterations表示迭代次数
    dest = cv2.erode(lena, kernel, iterations=1)

    # 现实图片
    cv2.imshow('Erosion', dest)
    cv2.imshow('Normal', lena)
    cv2.waitKey(0)
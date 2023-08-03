import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
累计概率霍夫变换
'''
if __name__ == '__main__':
    rows = 2
    columns = 2

    circles_img = cv2.imread('data/circles.png')
    plt.subplot(rows, columns, 1)
    plt.title('Circles-IMG')
    plt.imshow(cv2.cvtColor(circles_img.copy(), cv2.COLOR_BGR2RGB))

    # 第一步：把图像转化为灰度图像
    circles_gray = cv2.cvtColor(circles_img, cv2.COLOR_BGR2GRAY)
    plt.subplot(rows, columns, 2)
    plt.title('Circles-Gray')
    plt.imshow(cv2.cvtColor(circles_gray.copy(), cv2.COLOR_BGR2RGB))

    # 第二步：对图像进行降噪
    circles_blur = cv2.GaussianBlur(circles_gray, (3, 3), 2)
    plt.subplot(rows, columns, 3)
    plt.title('Circles-Blur')
    plt.imshow(cv2.cvtColor(circles_blur.copy(), cv2.COLOR_BGR2RGB))

    '''
    累计概率霍夫变换
    method：检测方法
    dp：累加器图像与输入图像分辨率的倒数
    minDist：圆心之间的最小距离
    return：[(x, y, r), ..., (x, y, r)]
    '''
    circles = cv2.HoughCircles(circles_blur, cv2.HOUGH_GRADIENT, 1.5, 10, param1=100, param2=100, maxRadius=100)[0]
    for x, y, r in circles:
        # 绘制圆
        cv2.circle(circles_img, (x, y), r, (255, 0, 0), thickness=5)

    print('Circles Size = ', len(circles))

    plt.subplot(rows, columns, 4)
    plt.title('Circles')
    plt.imshow(cv2.cvtColor(circles_img.copy(), cv2.COLOR_BGR2RGB))

    plt.show()
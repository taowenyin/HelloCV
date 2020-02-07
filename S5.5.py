import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
霍夫变换
'''
if __name__ == '__main__':
    rows = 2
    columns = 3

    building = cv2.imread('data/building.png')
    plt.subplot(rows, columns, 1)
    plt.title('Building')
    plt.imshow(cv2.cvtColor(building.copy(), cv2.COLOR_BGR2RGB))

    # 第一步：把图像转化为灰度图像
    building_gray = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)
    plt.subplot(rows, columns, 2)
    plt.title('Building-Gray')
    plt.imshow(cv2.cvtColor(building_gray.copy(), cv2.COLOR_BGR2RGB))

    # 第二步：对图像进行降噪
    building_blur = cv2.blur(building_gray, (3, 3))
    plt.subplot(rows, columns, 3)
    plt.title('Building-Blur')
    plt.imshow(cv2.cvtColor(building_blur.copy(), cv2.COLOR_BGR2RGB))

    # 第三步：通过设置梯度大小和滞后阈值进行边缘检测
    building_canny = cv2.Canny(building_blur, 70, 150, apertureSize=3)
    plt.subplot(rows, columns, 4)
    plt.title('Building-Canny')
    plt.imshow(cv2.cvtColor(building_canny.copy(), cv2.COLOR_BGR2RGB))

    '''
    霍夫曼变换，并且把三维数组转为二维数组
    rho：以像素为单位的距离精度
    theta：以弧度为单位的角度精度
    threshold：累加平面的阈值参数
    return：[(距离, 弧度), (距离, 弧度), ..., (距离, 弧度)]
    '''
    lines = cv2.HoughLines(building_canny, 1, np.pi / 180, 150)[:, 0, :]
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # 绘制直线
        cv2.line(building, (x1, y1), (x2, y2), (255, 0, 0), thickness=3, lineType=8)

    plt.subplot(rows, columns, 5)
    plt.title('Building-Lines')
    plt.imshow(cv2.cvtColor(building.copy(), cv2.COLOR_BGR2RGB))

    plt.show()
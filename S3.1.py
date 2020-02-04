import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 读取图片
    lena = cv2.imread('data/Lena.png')
    # 把OpenCV的BGR转化为RBG
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)

    '''
    绘制椭圆
    img：原图像
    center：椭圆中心点坐标
    axes：椭圆长短轴大小
    angle：顺时针旋转角度
    startAngle：起始角度
    endAngle：结束角度
    color：颜色（BGR）
    thickness：线粗细
    lineType：线类型
    '''
    ellipse = lena.copy()
    cv2.ellipse(
        ellipse,
        (int(lena.shape[0] / 2), int(lena.shape[1] / 2)),
        (int(lena.shape[0] / 4), int(lena.shape[1] / 16)),
        0, 0, 360,
        (255, 255, 255),
        thickness=3, lineType=8)
    plt.subplot(1, 3, 1)
    plt.title('Ellipse')
    plt.imshow(ellipse)

    '''
    绘制圆形
    img：原图像
    center：圆形中心点坐标
    radius：圆形半径
    color：颜色（BGR）
    thickness：线粗细
    lineType：线类型
    '''
    circle = lena.copy()
    cv2.circle(circle,
               (int(lena.shape[0] / 2), int(lena.shape[1] / 2)),
               50,
               (255, 255, 255),
               thickness=3, lineType=8)
    plt.subplot(1, 3, 2)
    plt.title('Circle')
    plt.imshow(circle)

    '''
    绘制直线
    img：原图像
    pt1：起点坐标
    pt2：终点半径
    color：颜色（BGR）
    thickness：线粗细
    lineType：线类型
    '''
    line = lena.copy()
    cv2.line(line,
             (0, 0),
             (lena.shape[0], lena.shape[1]),
             (255, 255, 255),
             thickness=3, lineType=8)
    plt.subplot(1, 3, 3)
    plt.title('Line')
    plt.imshow(line)

    plt.show()

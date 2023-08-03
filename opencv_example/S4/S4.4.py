import cv2
import numpy as np
import matplotlib.pyplot as plt


# 开运算，闭运算，形态学梯度，顶帽，黑帽
if __name__ == '__main__':
    rows = 2
    columns = 3

    lena = cv2.imread('data/Lena.png')
    # lena = cv2.imread('data/morphology.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    # 创建一个卷积核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    print(kernel)

    '''
    开运算：先腐烛再膨胀，消除小物体
    ksize：巻积核大小
    iterations：迭代次数
    '''
    open = cv2.morphologyEx(lena, cv2.MORPH_OPEN, kernel, iterations=1)
    plt.subplot(rows, columns, 2)
    plt.title('Open')
    plt.imshow(cv2.cvtColor(open, cv2.COLOR_BGR2RGB))

    '''
    闭运算：先膨胀再腐烛，消除小黑洞
    ksize：巻积核大小
    iterations：迭代次数
    '''
    close = cv2.morphologyEx(lena, cv2.MORPH_CLOSE, kernel, iterations=1)
    plt.subplot(rows, columns, 3)
    plt.title('Close')
    plt.imshow(cv2.cvtColor(close, cv2.COLOR_BGR2RGB))

    '''
    形态学梯度：膨胀-腐烛，团块的边缘突出
    ksize：巻积核大小
    iterations：迭代次数
    '''
    gradient = cv2.morphologyEx(lena, cv2.MORPH_GRADIENT, kernel, iterations=1)
    plt.subplot(rows, columns, 4)
    plt.title('Gradient')
    plt.imshow(cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB))

    '''
    顶帽运算：原图-开运算，放大局部裂痕或低亮度区域
    ksize：巻积核大小
    iterations：迭代次数
    '''
    tophat = cv2.morphologyEx(lena, cv2.MORPH_TOPHAT, kernel, iterations=1)
    plt.subplot(rows, columns, 5)
    plt.title('Top-Hat')
    plt.imshow(cv2.cvtColor(tophat, cv2.COLOR_BGR2RGB))

    '''
    黑帽运算：闭运算-原图，突出轮廓周围更暗区域
    ksize：巻积核大小
    iterations：迭代次数
    '''
    blackhat = cv2.morphologyEx(lena, cv2.MORPH_BLACKHAT, kernel, iterations=1)
    plt.subplot(rows, columns, 6)
    plt.title('Black-Hat')
    plt.imshow(cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB))

    plt.show()
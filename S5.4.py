import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
scharr算子
'''
if __name__ == '__main__':
    rows = 2
    columns = 2

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena.copy(), cv2.COLOR_BGR2RGB))

    scharrX = cv2.Scharr(lena, cv2.CV_32F, 1, 0)
    plt.subplot(rows, columns, 2)
    plt.title('Scharr-X')
    plt.imshow(cv2.cvtColor(scharrX.copy(), cv2.COLOR_BGR2RGB))

    scharrY = cv2.Scharr(lena, cv2.CV_32F, 0, 1)
    plt.subplot(rows, columns, 3)
    plt.title('Scharr-Y')
    plt.imshow(cv2.cvtColor(scharrY.copy(), cv2.COLOR_BGR2RGB))

    scharr = cv2.addWeighted(scharrX, 0.5, scharrY, 0.5, 0)
    plt.subplot(rows, columns, 4)
    plt.title('Scharr')
    plt.imshow(cv2.cvtColor(scharr.copy(), cv2.COLOR_BGR2RGB))

    plt.show()
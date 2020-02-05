import cv2
import numpy as np
import matplotlib.pyplot as plt


# 漫水填充
if __name__ == '__main__':
    rows = 1
    columns = 2

    lena = cv2.imread('data/Lena.png')
    plt.subplot(rows, columns, 1)
    plt.title('Lena')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    # 构建掩模，尺寸比图像大2个像素
    mask = np.zeros([lena.shape[0] + 2, lena.shape[1] + 2], np.uint8)
    '''
    满水算法
    mask：掩模
    seedPoint：起始点
    newVal：要填充的颜色
    loDiff：下界的差值
    upDiff：上界的差值
    flags：FLOODFILL_FIXED_RANGE表示使用填充色填充，FLOODFILL_MASK_ONLY表示不是用填充色，而使用对应掩码
    '''
    data = cv2.floodFill(lena, mask, (250, 100), (0, 255, 255), (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    plt.subplot(rows, columns, 2)
    plt.title('Flood')
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

    plt.show()
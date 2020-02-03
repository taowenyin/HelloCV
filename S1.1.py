import cv2


'''
OpenCV的安装使用PIP不要使用conda安装
命令：
pip install opencv-python
pip install opencv-contrib-python
'''
if __name__ == '__main__':
    # 读取图片
    lena = cv2.imread('data/Lena.png')
    # 现实图片
    cv2.imshow('Normal', lena)
    cv2.waitKey(0)
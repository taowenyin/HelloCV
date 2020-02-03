import cv2


# OpenCV读取视频
if __name__ == '__main__':
    # 创建一个实例
    cap = cv2.VideoCapture(0)
    isRunning = True

    # 打开视频
    cap.open('data/test.mp4')

    while isRunning:
        # ret表示读取成功，frame表示读取到的视频帧
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera', frame)
        # 得到按键
        key = cv2.waitKey(30)
        if key == ord('q'):
            isRunning = False

    # 释放VideoCapture
    cap.release()
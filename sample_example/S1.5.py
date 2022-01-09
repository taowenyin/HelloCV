import cv2


# OpenCV读取摄像头
if __name__ == '__main__':
    # 创建一个实例
    cap = cv2.VideoCapture(0)
    isRunning = True

    print('摄像头是否已经打开？{}', format(cap.isOpened()))

    # 设置摄像头图像的大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
import cv2


# 保存摄像头图片
if __name__ == '__main__':
    # 获取当前OpenCV的版本
    print('CV Version', cv2.getVersionString())

    cap = cv2.VideoCapture(0)
    isRunning = True

    print('摄像头是否已经打开？{}', format(cap.isOpened()))

    # 设置摄像头图像的大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while isRunning:
        # ret表示读取成功，frame表示读取到的视频帧
        ret, frame = cap.read()

        # 判断摄像头数据错误
        if not ret:
            print("图像获取失败，请按照说明进行问题排查")
            break

        cv2.imshow('Camera', frame)
        # 得到按键
        key = cv2.waitKey(30)

        # 退出程序
        if key == ord('q'):
            isRunning = False
        # 截图
        if key == ord('c'):
            print('图像信息', frame)
            cv2.imwrite('capture.png', frame)
            print('保存图片成功')

    # 释放VideoCapture
    cap.release()
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 形状匹配
if __name__ == "__main__":
    rows = 1
    columns = 3

    cs1 = cv2.imread("data/hand.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(cs1.copy(), cv2.COLOR_BGR2RGB))

    # 显示凸包
    src = cv2.imread("data/hand.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(contours[0])
    cv2.polylines(src, [hull], True, (0, 255, 0), 2)
    plt.subplot(rows, columns, 2)
    plt.title("Hull")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    # 显示凸缺陷
    src = cv2.imread("data/hand.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        cv2.line(src, start, end, [0, 0, 255], 2)
        cv2.circle(src, far, 5, [255, 0, 0], -1)
    plt.subplot(rows, columns, 3)
    plt.title("Defects")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    plt.show()

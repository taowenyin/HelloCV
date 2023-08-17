import cv2
import numpy as np
import matplotlib.pyplot as plt


# 形状匹配
if __name__ == "__main__":
    rows = 1
    columns = 3

    cs1 = cv2.imread("data/cs1.png")
    plt.subplot(rows, columns, 1)
    plt.title("CS1")
    plt.imshow(cv2.cvtColor(cs1.copy(), cv2.COLOR_BGR2RGB))

    cs2 = cv2.imread("data/cs2.png")
    plt.subplot(rows, columns, 2)
    plt.title("CS2")
    plt.imshow(cv2.cvtColor(cs2.copy(), cv2.COLOR_BGR2RGB))

    cs3 = cv2.imread("data/cs3.png")
    plt.subplot(rows, columns, 3)
    plt.title("CS3")
    plt.imshow(cv2.cvtColor(cs3.copy(), cv2.COLOR_BGR2RGB))

    gray1 = cv2.cvtColor(cs1.copy(), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cs2.copy(), cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(cs3.copy(), cv2.COLOR_BGR2GRAY)

    ret, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    ret, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    ret, binary3 = cv2.threshold(gray3, 127, 255, cv2.THRESH_BINARY)

    contours1, hierarchy = cv2.findContours(
        binary1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours2, hierarchy = cv2.findContours(
        binary2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours3, hierarchy = cv2.findContours(
        binary3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    cnt1 = contours1[0]
    cnt2 = contours2[0]
    cnt3 = contours3[0]
    ret1 = cv2.matchShapes(cnt1, cnt1, cv2.CONTOURS_MATCH_I1, 0)
    ret2 = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0)
    ret3 = cv2.matchShapes(cnt1, cnt3, cv2.CONTOURS_MATCH_I1, 0)

    print("cs1与cs1的匹配程度 = {}".format(ret1))
    print("cs1与cs2的匹配程度 = {}".format(ret2))
    print("cs1与cs3的匹配程度 = {}".format(ret3))

    plt.show()

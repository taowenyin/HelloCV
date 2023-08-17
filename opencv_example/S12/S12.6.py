import cv2
import numpy as np
import matplotlib.pyplot as plt


# 轮廓拟合
if __name__ == "__main__":
    rows = 2
    columns = 3

    src = cv2.imread("data/cs1.png")
    rect_src = src.copy()
    circle_src = src.copy()
    ellipse_src = src.copy()
    line_src = src.copy()
    triangle_src = src.copy()

    plt.subplot(rows, columns, 1)
    plt.title("CS1")
    plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # 矩形
    b_x, b_y, b_w, b_h = cv2.boundingRect(contours[0])
    # 圆形
    (c_x, c_y), radius = cv2.minEnclosingCircle(contours[0])
    # 椭圆
    ellipse = cv2.fitEllipse(contours[0])
    # 直线
    lx, ly, x, y = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
    # 三角形
    area, trgl = cv2.minEnclosingTriangle(contours[0])

    # 绘制矩形
    brcnt = np.array(
        [[[b_x, b_y]], [[b_x + b_w, b_y]], [[b_x + b_w, b_y + b_h]], [[b_x, b_y + b_h]]]
    )
    cv2.drawContours(rect_src, [brcnt], -1, (255, 255, 255), 2)
    # 绘制圆形
    cv2.circle(circle_src, (int(c_x), int(c_y)), int(radius), (255, 255, 255), 2)
    # 绘制椭圆
    cv2.ellipse(ellipse_src, ellipse, (0, 255, 0), 3)
    # 绘制直线
    h, w, c = line_src.shape
    left_y = int((-x * ly / lx) + y)
    right_y = int(((w - x) * ly / lx) + y)
    cv2.line(line_src, (w - 1, right_y), (0, left_y), (0, 255, 0), 2)
    # 绘制三角形
    for i in range(0, 3):
        cv2.line(
            triangle_src,
            (int(trgl[i][0][0]), int(trgl[i][0][1])),
            (int(trgl[(i + 1) % 3][0][0]), int(trgl[(i + 1) % 3][0][1])),
            (255, 255, 255),
            2,
        )

    plt.subplot(rows, columns, 2)
    plt.title("Rect Bounding")
    plt.imshow(cv2.cvtColor(rect_src, cv2.COLOR_BGR2RGB))

    plt.subplot(rows, columns, 3)
    plt.title("Circle Bounding")
    plt.imshow(cv2.cvtColor(circle_src, cv2.COLOR_BGR2RGB))

    plt.subplot(rows, columns, 4)
    plt.title("Ellipse Bounding")
    plt.imshow(cv2.cvtColor(ellipse_src, cv2.COLOR_BGR2RGB))

    plt.subplot(rows, columns, 5)
    plt.title("Line Bounding")
    plt.imshow(cv2.cvtColor(line_src, cv2.COLOR_BGR2RGB))

    plt.subplot(rows, columns, 6)
    plt.title("Triangle Bounding")
    plt.imshow(cv2.cvtColor(triangle_src, cv2.COLOR_BGR2RGB))

    plt.show()

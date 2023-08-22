import cv2
import numpy as np
import matplotlib.pyplot as plt


# 模板匹配
if __name__ == "__main__":
    rows = 1
    columns = 3

    src = cv2.imread("data/computer.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/computer.png")
    gray = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    plt.subplot(rows, columns, 2)
    plt.title("Edge")
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/computer.png")
    gray = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 160, minLineLength=100, maxLineGap=10
    )
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(src, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.subplot(rows, columns, 3)
    plt.title("HoughLinesP")
    plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    plt.show()

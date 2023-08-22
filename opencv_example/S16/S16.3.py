import cv2
import numpy as np
import matplotlib.pyplot as plt


# 模板匹配
if __name__ == "__main__":
    rows = 1
    columns = 3

    src = cv2.imread("data/chess.png")
    plt.subplot(rows, columns, 1)
    plt.title("Chess")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/chess.png")
    gray = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150, apertureSize=3)
    plt.subplot(rows, columns, 2)
    plt.title("Edge")
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/chess.png", cv2.IMREAD_COLOR)
    gray = cv2.imread("data/chess.png", cv2.IMREAD_GRAYSCALE)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        300,
        param1=150,
        param2=60,
        minRadius=100,
        maxRadius=200,
    )
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        cv2.circle(src, (circle[0], circle[1]), circle[2], (255, 0, 0), 12)
        cv2.circle(src, (circle[0], circle[1]), 2, (255, 0, 0), 12)
    plt.subplot(rows, columns, 3)
    plt.title("HoughCircles")
    plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    plt.show()

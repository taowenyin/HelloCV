import cv2
import numpy as np
import matplotlib.pyplot as plt


# 模板匹配
if __name__ == "__main__":
    rows = 2
    columns = 2

    src = cv2.imread("data/胶囊.png")
    plt.subplot(rows, columns, 1)
    plt.title("Org")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    temp = cv2.imread("data/胶囊Temp.png")
    h, w, c = temp.shape
    plt.subplot(rows, columns, 2)
    plt.title("Temp")
    plt.imshow(cv2.cvtColor(temp.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/胶囊.png")
    temp = cv2.imread("data/胶囊Temp.png")
    # 使用TM_CCORR匹配
    matchValue = cv2.matchTemplate(src, temp, cv2.TM_CCORR_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchValue)
    # 使用TM_CCORR匹配时，取最大值
    topLeft = maxLoc
    bottomRight = (topLeft[0] + w, topLeft[1] + h)
    # 画矩形
    cv2.rectangle(src, topLeft, bottomRight, 255, 2)
    plt.subplot(rows, columns, 3)
    plt.title("Match")
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    src = cv2.imread("data/胶囊.png")
    temp = cv2.imread("data/胶囊Temp.png")
    draw_count = 0
    # 使用TM_CCORR匹配
    for r in range(0, 360):
        # 自适应调整图像的尺寸
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), r, 1.0)
        cos_theta = np.abs(rotation_matrix[0, 0])
        sin_theta = np.abs(rotation_matrix[0, 1])
        new_width = int((h * sin_theta) + (w * cos_theta))
        new_height = int((h * cos_theta) + (w * sin_theta))
        rotation_matrix[0, 2] += (new_width / 2) - w // 2
        rotation_matrix[1, 2] += (new_height / 2) - h // 2
        rotated_image = cv2.warpAffine(temp, rotation_matrix, (new_width, new_height))

        # 匹配模板
        matchValue = cv2.matchTemplate(src, rotated_image, cv2.TM_CCORR_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchValue)

        if maxVal < 0.93:
            continue
        draw_count += 1
        # 使用TM_CCORR匹配时，取最大值
        topLeft = maxLoc
        bottomRight = (topLeft[0] + w, topLeft[1] + h)
        # 画矩形
        cv2.rectangle(src, topLeft, bottomRight, 255, 2)
    plt.subplot(rows, columns, 4)
    plt.title("Multi Match: {}".format(draw_count))
    plt.imshow(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB))

    plt.show()

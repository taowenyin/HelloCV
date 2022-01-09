import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 使用ORB或SIFT进行关键点检测
def keypoints_detect(image, detecter):
    # 把图像转化为灰度图
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if detecter == 'SIFT':
        # 创建SIFT对象
        detect = cv.SIFT_create()
    else:
        # 创建ORB对象
        detect = cv.ORB().create()

    # keypoints:特征点向量,向量内的每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息(角度、关键特征点坐标等)
    # features:表示输出的sift特征向量，通常是128维的
    keypoints, features = detect.detectAndCompute(image, None)

    # cv.drawKeyPoints():在图像的关键特征点部位绘制一个小圆圈
    # 如果传递标志flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,它将绘制一个大小为keypoint的圆圈并显示它的方向
    # 这种方法同时显示图像的坐标，大小和方向，是最能显示特征的一种绘制方式
    keypoints_image = cv.drawKeypoints(gray_image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    return keypoints_image, keypoints, features


# 使用KNN检测来自左右图像的SIFT特征，随后进行匹配
def get_feature_point_ensemble(query_image, train_image):
    # 创建匹配器
    matcher = cv.BFMatcher()

    # knnMatch()函数：返回每个特征点的最佳匹配k个匹配点，features_right为匹配原图，features_left为匹配目标图
    match_points = matcher.knnMatch(query_image, train_image, k=2)

    # 利用sorted()函数对matches对象进行升序(默认)操作，
    # x:x[]字母可以随意修改，排序方式按照中括号[]里面的维度进行排序
    match_points = sorted(match_points, key=lambda x: x[0].distance / x[1].distance)

    # 建立列表good_match用于存储匹配的点集
    good_match = []
    for best_1, best_2 in match_points:
        # ratio的值越大，匹配的线条越密集，但错误匹配点也会增多
        ratio = 0.6

        if best_1.distance < ratio * best_2.distance:
            good_match.append(best_1)

    return good_match


def panorama_stitching(query_image, train_image, detecter):
    _, keypoints_right, features_right = keypoints_detect(train_image, detecter)
    _, keypoints_left, features_left = keypoints_detect(query_image, detecter)

    good_match = get_feature_point_ensemble(features_right, features_left)

    if len(good_match) > 4:
        # 获取匹配对的点坐标
        pts_r = np.float32([keypoints_right[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        pts_l = np.float32([keypoints_left[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

        # ransac_reproj_Threshold：将点对视为内点的最大允许重投影错误阈值(仅用于RANSAC和RHO方法时),
        # 若srcPoints和dstPoints是以像素为单位的，该参数通常设置在1到10的范围内
        ransac_reproj_Threshold = 4

        # 第四步：图像配准
        # cv.findHomography():计算多个二维点对之间的最优单映射变换矩阵 H(3行x3列),使用最小均方误差或者RANSAC方法
        # 函数作用:利用基于RANSAC的鲁棒算法选择最优的四组配对点，再计算转换矩阵H(3*3)并返回,以便于反向投影错误率达到最小
        homography, status = cv.findHomography(pts_r, pts_l, cv.RANSAC, ransac_reproj_Threshold)

        # 第五步：投影计算
        # 第六步：拼缝计算
        # cv.warpPerspective()：透视变换函数，用于解决cv2.warpAffine()不能处理视场和图像不平行的问题
        # 作用：就是对图像进行透视变换，可保持直线不变形，但是平行线可能不再平行
        panorama = cv.warpPerspective(train_image, homography,
                                      (train_image.shape[1] + query_image.shape[1], train_image.shape[0]))
        right_warp_img = panorama.copy()

        # 第七步：图像融合
        # 将左图加入到变换后的右图像的左端即获得最终图像
        panorama[0: query_image.shape[0], 0: query_image.shape[1]] = query_image

        return panorama, right_warp_img


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 第一步：输入图像
    image_right = cv.cvtColor(cv.imread('data/right.jpg'), cv.COLOR_BGR2RGB)
    image_left = cv.cvtColor(cv.imread('data/left.jpg'), cv.COLOR_BGR2RGB)

    # 第二步：特征点提取
    # 获取检测到关键特征点后的图像的相关参数
    sift_keypoints_image_left, sift_keypoints_left, sift_features_left = keypoints_detect(image_left,
                                                                                          detecter='SIFT')
    sift_keypoints_image_right, sift_keypoints_right, sift_features_right = keypoints_detect(image_right,
                                                                                             detecter='SIFT')

    orb_keypoints_image_left, orb_keypoints_left, orb_features_left = keypoints_detect(image_left,
                                                                                       detecter='ORB')
    orb_keypoints_image_right, orb_keypoints_right, orb_features_right = keypoints_detect(image_right,
                                                                                          detecter='ORB')

    # 第三步：特征点匹配
    # ORB获取最佳匹配
    orb_good_match = get_feature_point_ensemble(orb_features_right, orb_features_left)
    # SIFT获取最佳匹配
    sift_good_match = get_feature_point_ensemble(sift_features_right, sift_features_left)

    # cv.drawMatches():在提取两幅图像特征之后，画出匹配点对连线
    # matchColor – 匹配的颜色（特征点和连线),若matchColor==Scalar::all(-1),颜色随机
    all_sift_good_match_image = cv.drawMatches(image_right, sift_keypoints_right, image_left, sift_keypoints_left,
                                               sift_good_match, None, None, None, None,
                                               flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    all_orb_good_match_image = cv.drawMatches(image_right, orb_keypoints_right, image_left, orb_keypoints_left,
                                              orb_good_match, None, None, None, None,
                                              flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 第八步：生成全景图
    sift_panorama, sift_right_warp_img = panorama_stitching(image_right, image_left, detecter='SIFT')
    orb_panorama, orb_right_warp_img = panorama_stitching(image_right, image_left, detecter='ORB')

    # =========================画图===================================
    rows = 3
    cols = 3

    plt.subplot(rows, cols, 1)
    plt.title('SIFT-左侧图像的关键点')
    # 利用np.hstack()函数同时将原图和绘有关键特征点的图像沿着竖直方向(水平顺序)堆叠起来
    plt.imshow(np.hstack((image_left, sift_keypoints_image_left)))
    plt.subplot(rows, cols, 2)
    plt.title('SIFT-关键点匹配')
    plt.imshow(all_sift_good_match_image)
    plt.subplot(rows, cols, 3)
    plt.title('SIFT-扭曲变换后的右图')
    plt.imshow(sift_right_warp_img)
    plt.subplot(rows, cols, 4)
    plt.title('SIFT-全景图')
    plt.imshow(sift_panorama)

    plt.subplot(rows, cols, 6)
    plt.title('ORB-左侧图像的关键点')
    # 利用np.hstack()函数同时将原图和绘有关键特征点的图像沿着竖直方向(水平顺序)堆叠起来
    plt.imshow(np.hstack((image_left, orb_keypoints_image_left)))
    plt.subplot(rows, cols, 7)
    plt.title('ORB-关键点匹配')
    plt.imshow(all_orb_good_match_image)
    plt.subplot(rows, cols, 8)
    plt.title('ORB-扭曲变换后的右图')
    plt.imshow(orb_right_warp_img)
    plt.subplot(rows, cols, 9)
    plt.title('ORB-全景图')
    plt.imshow(orb_panorama)

    plt.show()


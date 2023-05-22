import cv2
import numpy as np


def overlap_area(poly1, poly2):
    # 将两个矩形坐标按照顺时针顺序存储

    # poly1 = np.array([[rect1[0], rect1[1]], [rect1[0], rect1[3]], [rect1[2], rect1[3]], [rect1[2], rect1[1]]], dtype=np.float32)
    # poly2 = np.array([[rect2[0], rect2[1]], [rect2[0], rect2[3]], [rect2[2], rect2[3]], [rect2[2], rect2[1]]], dtype=np.float32)

    # print("poly1:__________________")
    # print(poly1)
    # print("poly2:__________________")
    # print(poly2)
    # 计算两个凸多边形是否有交集
    retval, intersection = cv2.intersectConvexConvex(poly1, poly2)

    # 如果有交集，返回交集面积
    if retval:
        area = cv2.contourArea(intersection)
        return area
    else:
        return 0



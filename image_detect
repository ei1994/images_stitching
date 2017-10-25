"""
特征匹配及查找对象
"""

import cv2
import numpy as np
import matplotlib.pylab as plt

MIN_MATCH_COUNT = 10
img1 = cv2.imread('images/111.jpg')          # queryImage
img2 = cv2.imread('images/11.jpg')          # trainImage

def SIFT():
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
          good.append(m)
          
    if len(good)>MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
       
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#        wrap = cv2.warpPerspective(img2, H, (img2.shape[1]+img2.shape[1] , img2.shape[0]+img2.shape[0]))
#        wrap[0:img2.shape[0], 0:img2.shape[1]] = img1
        
        matchesMask = mask.ravel().tolist()
#        # 获得原图像的高和宽
        h,w,_ = img1.shape
#        # 使用得到的变换矩阵对原图像的四个角进行变换,获得在目标图像上对应的坐标。
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        # 在原图中画出目标所在位置框, cv2.LINE_AA表示闭合框
        cv2.polylines(img2,[np.int32(dst)],True,255,10, cv2.LINE_AA)
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        
#    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#    singlePointColor = None,
#    matchesMask = matchesMask, # draw only inliers
#    flags = 2)
#    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
##    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,matchesMask = matchesMask,flags=2)
    
    good = np.expand_dims(good,1)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:20],None, flags=2)
    return img3
    
if __name__ == '__main__':
    result = SIFT()
    cv2.imshow('result1.jpg',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

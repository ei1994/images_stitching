#!/usr/bin/env python3
# -*- coding: utf-8 -*-
‘’‘
图像拼接有两种拼接方法，
一是以第一张图为基准，变换第二张图与第一张拼接；
二是以第二张图为基准，变换第一张图与第二张拼接。
’‘’
import cv2
import numpy as np
import argparse

class Stitcher:
    def __init__(self):
        self.descriptor = cv2.xfeatures2d.SIFT_create()
    
    def match_keypoints(self, featuresA, featuresB):
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(featuresA, featuresB, k=2)
                
        return raw_matches
        
    def compute_H(self, raw_matches, raw_kpsA, raw_kpsB, ratio, reprojThresh):
        good = []
        # loop for finding good matches
        for m,n in raw_matches:
            if m.distance < ratio*n.distance:
                good.append(( m.trainIdx, m.queryIdx))
        if len(good) > 4:
            kpsA = np.float32(
                    [raw_kpsA[i].pt for (_, i) in good])
            kpsB = np.float32(
                    [raw_kpsB[i].pt for (i, _) in good])
            # compute the homography
            H, status = cv2.findHomography(kpsB, kpsA, cv2.RANSAC, reprojThresh)
            
            return (good, H, status)
        return None
        
    def draw_matches(self, raw_matches, img1, raw_kps1, img2, raw_kps2):
        good = []
        for m,n in raw_matches:
           if m.distance < 0.75*n.distance:
              good.append([m])
        result = cv2.drawMatchesKnn(img1,raw_kps1,img2,raw_kps2,good[0:15],None, flags=2)
        
        return result
        
    # 以左图为基准，变换右图来拼接         
    def stitch_left_based(self, img1, img2, H):
        wrap = cv2.warpPerspective(img2, H, (100+img1.shape[1]+img2.shape[1] , 100+img2.shape[0]))
        wrap[0:img2.shape[0], 0:img2.shape[1]] = img1
    #    wrap[0:img1.shape[0], img1.shape[1]:] = img2
            
        return wrap
        
    # 以右图为基准，变换左图来拼接    
    def stitch_right_based(self, img1, img2, H):
        xh = np.linalg.inv(H)
        f1 = np.dot(xh, np.array([0,0,1]))
        f1 = f1/f1[-1]
        xh[0][-1] += abs(f1[0])
        xh[1][-1] += abs(f1[1])
        offsety = abs(int(f1[1]))
        offsetx = abs(int(f1[0]))
        wrap = cv2.warpPerspective(img1, xh, (img1.shape[1]+img2.shape[1] ,img2.shape[0]+img2.shape[0]))
        wrap[offsety:img2.shape[0]+offsety, offsetx:img2.shape[1]+offsetx] = img2
            
        return wrap
    
    def match(self, image1, image2, ratio = 0.75, reprojThresh =5.0):
        raw_kps1, features1 = self.descriptor.detectAndCompute(image1, None)
        raw_kps2, features2 = self.descriptor.detectAndCompute(image2, None)
        
        raw_matches = self.match_keypoints(features1, features2)
        # draw the lines in images
        result = self.draw_matches(raw_matches,image1,raw_kps1,image2,raw_kps2)
        good, H, status = self.compute_H(raw_matches, raw_kps1, raw_kps2, ratio, reprojThresh)
        final1 = self.stitch_left_based(image1, image2, H)
        rows, cols = np.where(final1[:,:,0] !=0)
    
        min_row, max_row = min(rows), max(rows) +1
        min_col, max_col = min(cols), max(cols) +1
        final = final1[min_row:max_row,min_col:max_col,:]#去除黑色无用部分
        return result, final
 
if __name__ == '__main__':
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", default='images/1.jpg',
    	help="path to the first image")
    ap.add_argument("-s", "--second", default='images/2.jpg',
    	help="path to the second image")      
    args = vars(ap.parse_args())
    
    ratio = 0.75
    reprojThresh =5.0
    image1 = cv2.imread(args["first"])
    image2 = cv2.imread(args["second"])
    stitcher = Stitcher()
    result, final = stitcher.match(image1, image2, ratio, reprojThresh)
    
    cv2.imshow('matches',result)
    cv2.imshow('stitch',final)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
cv2.waitKey(1)

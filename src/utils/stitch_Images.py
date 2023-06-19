#!/usr/bin/python

import os
import sys
import cv2
import math
import numpy as np
#import utils

from PIL import Image
from numpy import linalg

def filter_matches(matches, ratio = 0.75):
    filtered_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])
    
    return filtered_matches

def imageDistance(matches):

    sumDistance = 0.0

    for match in matches:

        sumDistance += match.distance

    return sumDistance

def findDimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0,0]
    base_p2[:2] = [x,0]
    base_p3[:2] = [0,y]
    base_p4[:2] = [x,y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

        if ( max_x == None or normal_pt[0,0] > max_x ):
            max_x = normal_pt[0,0]

        if ( max_y == None or normal_pt[1,0] > max_y ):
            max_y = normal_pt[1,0]

        if ( min_x == None or normal_pt[0,0] < min_x ):
            min_x = normal_pt[0,0]

        if ( min_y == None or normal_pt[1,0] < min_y ):
            min_y = normal_pt[1,0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)

def stitchImages(base_img_rgb, img2, H_inv):

    (min_x, min_y, max_x, max_y) = findDimensions(img2, H_inv)

    # Adjust max_x and max_y by base img size
    max_x = max(max_x, base_img_rgb.shape[1])
    max_y = max(max_y,base_img_rgb.shape[0])
    move_h = np.matrix(np.identity(3), np.float32)
    if ( min_x < 0 ):
            move_h[0,2] += -min_x
            max_x += -min_x
    if ( min_y < 0 ):
            move_h[1,2] += -min_y
            max_y += -min_y
    mod_inv_h = move_h * H_inv

    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

    # Warp the new image given the homography from the old image
    base_img_warp = cv2.warpPerspective(base_img_rgb, move_h, (img_w, img_h))

    next_img_warp = cv2.warpPerspective(img2, mod_inv_h, (img_w, img_h))
    #print "Warped next image"

    # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
    # cv2.destroyAllWindows()

    # Put the base image on an enlarged palette
    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
    final_img=np.zeros((img_h, img_w, 3), np.uint8)

    #print "Enlarged Image Shape: ", enlarged_base_img.shape
    #print "Base Image Shape: ", base_img_rgb.shape
    #print "Base Image Warp Shape: ", base_img_warp.shape

    # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
    # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp
    # Create a mask from the warped image for constructing masked composite
    img2_HSV = cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2HSV)
    data_map= cv2.inRange(img2_HSV, (0, 0, 30), (255, 255, 255))
    img3_HSV = cv2.cvtColor(base_img_warp, cv2.COLOR_BGR2HSV)

    data_map2 = cv2.inRange(img3_HSV, (0, 0, 30), (255, 255, 255))
    #mask=np.bitwise_not(data_map)
    im1 = cv2.bitwise_and(base_img_warp, base_img_warp, mask=data_map2 )

    im2 = cv2.bitwise_and(next_img_warp, next_img_warp, mask=data_map)

    masked1=cv2.distanceTransform(data_map2, cv2.DIST_L1, 5)
    cv2.normalize(masked1, masked1, 0, 1.0, cv2.NORM_MINMAX)
    masked2 = cv2.distanceTransform(data_map, cv2.DIST_L1, 5)
    cv2.normalize(masked2, masked2, 0, 1.0, cv2.NORM_MINMAX)
    _, dist = cv2.threshold(masked2, 0.02, 1.0, cv2.THRESH_BINARY)
    # Dilate a bit the dist image
    #kernel1 = np.ones((3, 3), dtype=np.uint8)
    #dist = cv2.dilate(masked2, kernel1)

    for i in range(0, img_h):
        for j in range(0, img_w):
            a = np.float32(masked1[i, j])
            b = np.float32(masked2[i, j])
            if a == b == 0:
                alpha = 0
            else:
                alpha = a / (a + b)
            if dist[i, j] != 0: final_img[i, j] = im2[i, j]
            else:
              final_img[i, j] = alpha * im1[i, j] + (1 - alpha) * im2[i, j]

    cv2.imwrite("image.jpg", final_img)
    return   final_img,mod_inv_h





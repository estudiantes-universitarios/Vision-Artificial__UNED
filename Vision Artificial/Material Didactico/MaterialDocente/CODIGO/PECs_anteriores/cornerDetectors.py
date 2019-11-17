#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
 

img = cv2.imread('imagenes/box.jpg',0)


# Initiate FAST object with default values
#fast = cv2.FastFeatureDetector()
fast = cv2.FastFeatureDetector_create(threshold=100)


# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
print ("Threshold: ", fast.getThreshold())
print ("nonmaxSuppression: ", fast.getNonmaxSuppression())
print ("neighborhood: ", fast.getType())
print ("Total Keypoints with nonmaxSuppression: ", len(kp))


cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print ("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

cv2.imshow('fast_false.png',img3)
cv2.waitKey(0)

cv2.destroyAllWindows()
print ("---- fin -----")







'''




img = cv2.imread('lena.png')
img_orig = copy.copy(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
#must give a float32 data type input
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
 
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
 
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.001*dst.max()]=[0,0,255]
 
cv2.imshow('Original',img_orig)
cv2.imshow('Harris Corner Detector',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    
   
    
    
    img = cv2.imread('box.jpg')
img_orig = copy.copy(img)
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
corners = cv2.goodFeaturesToTrack(grayimg,10,0.05,25)
corners = np.float32(corners)
 
for item in corners:
    x,y = item[0]
    cv2.circle(img,(x,y),5,255,-1)
 
cv2.imshow("Original",img_orig)
cv2.imshow("Top 10 corners",img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    
    
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:54:44 2019

@author: mrincon
"""

import numpy as np  
import cv2 

import matplotlib.pyplot as plt
import plot_cv_utils

import os
os.getcwd()


#%matplotlib inline
plt.rcParams['figure.figsize'] = (6.0, 4.5)




path_to_opencv_data = 'data_de_openCV/haarcascades/'
# (Windows) Anaconda\\Lib\\site-packages\\opencv....
# (Linux) Anaconda/Library/etc/haarcascades/

face_cascade = cv2.CascadeClassifier(path_to_opencv_data +
                             'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path_to_opencv_data +
                             'haarcascade_eye.xml')

img_cara = cv2.imread('figures/foto_carnet.png')
img_cara_gray = cv2.cvtColor(img_cara, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img_cara_gray, 1.3, 5) 

for (x,y,w,h) in faces:
    cv2.rectangle(img_cara,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = img_cara_gray[y:y+h, x:x+w]
    roi_color = img_cara[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

plot_cv_utils.plot_cv_img(img_cara)  






cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




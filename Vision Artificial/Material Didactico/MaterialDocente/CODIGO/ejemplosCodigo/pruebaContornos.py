#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:43:14 2019

@author: mrincon
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import toolsMRZ.utilsSegm as sg

image=cv2.imread('images/bw.png')

orig_image=image.copy()
cv2.imshow('original image',orig_image)
cv2.waitKey(1)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
ret, thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

plt.imshow(thresh)

cv2.imshow('thresholded image',thresh)
cv2.waitKey(1)


#_, contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
_, contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#_, contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
#_, contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_FLOODFILL,cv2.CHAIN_APPROX_NONE)



colores = ( (0,0,255), (0,255,0), (255,255,0), (255,0,255), (255,255,0), (125,125,0))
it=-1

for c in contours:
    it = it+1
    if it==6:
        it=0
    x,y,w,h=cv2.boundingRect(c)
    #cv2.drawContours(orig_image,[approx],0,(0,255,0),2)
    #cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
    #cv2.imshow('Bounding rect',orig_image)
    #cv2.waitKey(0)



    plt.plot(c[:,0,0], c[:,0,1],"g.-")
    plt.title("contorno")
    plt.show()
    #calculate accuracy as a percent of contour perimeter
    #accuracy=0.03*cv2.arcLength(c,True)
    #approx=cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(image,[c] ,0,colores[it],2)
    cv2.imshow('Approx polyDP', image)
    cv2.waitKey(1000)
    cv2.waitKey(10)
	
    kk=0
	
	
	

cv2.destroyAllWindows()


#plt.plot(contours[0][:,0,0], contours[0][:,0,1])






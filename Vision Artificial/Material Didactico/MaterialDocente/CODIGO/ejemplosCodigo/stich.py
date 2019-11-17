#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:15:11 2018

@author: mrincon
"""

from skimage.feature import ORB, match_descriptors
from skimage.io import imread
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.color import rgb2gray
from skimage.io import imsave, show
from skimage.color import gray2rgb
#from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform
import numpy as np
import cv2


def showPointsInImage(image, pointList):
    for p in pointList:
        P = ( int(p[1]), int(p[0]))
        cv2.circle(image, P, 4,(255,0,0))

    #T = tuple(map(tuple, pointList))
    #cv2.circle (img_aux, keypoints1[0,:]                            , 40, (255,0,0))
    #cv2.circle (img_aux  , (int(keypoints1[0,0]),int(keypoints1[0,1])), 40 , (255,0,0))




image0 = imread('images/goldengate1.png')
image0 = rgb2gray(image0)

image1 = imread('images/goldengate2.png')
image1 = rgb2gray(image1)


#inicialización de objeto ORB
orb = ORB(n_keypoints=1000, fast_threshold=0.05)
#orb = ORB(n_keypoints=50, fast_threshold=0.01)   
# pocos puntos implica menos posibilidades de establecer correspondencia con los de otra imagen con la que solo comparten un trozo.
# Si se sabe por dónde buscar la correspondencia, resulta más fácil restringir la zona de análisis en busca de puntos de interés.


# calcular puntos característicos
orb.detect_and_extract(image0)
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors

orb.detect_and_extract(image1)
keypoints2 = orb.keypoints
descriptors2 = orb.descriptors


# MOSTRAR RESULTADOS
#img_aux = np.zeros(image0.shape[0]*image0.shape[1]*3).reshape(image0.shape[0],image0.shape[1],3)
#img_aux[:,:,0]=img_aux[:,:,1]=img_aux[:,:,2]=image0.copy()
img_aux = image0.copy()
showPointsInImage(img_aux, keypoints1)
cv2.imshow("vent0",img_aux)
key = cv2.waitKey(0)

img_aux = image1.copy()
showPointsInImage(img_aux, keypoints2)
cv2.imshow("vent1",img_aux)
key = cv2.waitKey(0)


# establecer correspondencia. 
# Utilizala distancia de Hamming para medir la distancia entre descriptores
# Los descriptores de ORB son binarios
matches12 = match_descriptors(descriptors1,
                              descriptors2,
                              cross_check=True)

# generar modelo de la transformación
src = keypoints2[matches12[:, 1]][:, ::-1]                  # [:, ::-1] cambia el orden en las columnas
dst = keypoints1[matches12[:, 0]][:, ::-1]
transform_model, inliers =  ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=2)

r, c = image1.shape[:2]    # se coge la imagen 2 para calcular los nuevos límites de la imagen cosida

corners = np.array([[0, 0],
                    [0, r],
                    [c, 0],
                    [c, r]])

warped_corners = transform_model(corners)

all_corners = np.vstack((warped_corners, corners))

corner_min = np.min(all_corners, axis=0)
corner_max = np.max(all_corners, axis=0)

output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1])

offset = SimilarityTransform(translation=-corner_min)

image0_warp = warp(image0, offset.inverse, output_shape=output_shape, cval=-1)  # la primera imagen no se deforma, solo se traslada según el cálculo de la imagen completa realizado

#image1_warp = warp(image1, (model_robust + offset).inverse,output_shape=output_shape, cval=-1)
image1_warp = warp(image1, (transform_model + offset).inverse, output_shape=output_shape, cval=-1) # la segunda imagen se moldea de acuerdo con el punto de vista de la primera

# 
image0_mask = (image0_warp != -1)
image0_warp[~image0_mask] = 0
image0_alpha = np.dstack((gray2rgb(image0_warp), image0_mask))
 

image1_mask = (image1_warp != -1)
image1_warp[~image1_mask] = 0
image1_alpha = np.dstack((gray2rgb(image1_warp), image1_mask))


merged = (image0_alpha + image1_alpha)

alpha = merged[..., 3]
merged /= np.maximum(alpha, 1)[..., np.newaxis]

cv2.imshow('merged view', merged)
imsave('output.jpg', np.uint8(merged[:,:,:3]*255))






'''


# ---------- ----------------------


import argparse
import cv2, random

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/home/mrincon/data_examples/Lena.png', help='Image path.')
params = parser.parse_args()
image = cv2.imread(params.path)
w, h = image.shape[1], image.shape[0]


def rand_pt(mult=1.):
    return (random.randrange(int(w*mult)),
            random.randrange(int(h*mult)))

cv2.circle(image, rand_pt(), 40, (255, 0, 0))
cv2.circle(image, rand_pt(), 5, (255, 0, 0), cv2.FILLED)
cv2.circle(image, rand_pt(), 40, (255, 85, 85), 2)
cv2.circle(image, rand_pt(), 40, (255, 170, 170), 2, cv2.LINE_AA)
cv2.line(image, rand_pt(), rand_pt(), (0, 255, 0))
cv2.line(image, rand_pt(), rand_pt(), (85, 255, 85), 3)
cv2.line(image, rand_pt(), rand_pt(), (170, 255, 170), 3, cv2.LINE_AA)
cv2.arrowedLine(image, rand_pt(), rand_pt(), (0, 0, 255), 3, cv2.LINE_AA)
cv2.rectangle(image, rand_pt(), rand_pt(), (255, 255, 0), 3)
cv2.ellipse(image, rand_pt(), rand_pt(0.3), random.randrange(360), 0, 360, (255, 255, 255), 3)
cv2.putText(image, 'OpenCV', rand_pt(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

cv2.imshow("result", image)
key = cv2.waitKey(0)

'''


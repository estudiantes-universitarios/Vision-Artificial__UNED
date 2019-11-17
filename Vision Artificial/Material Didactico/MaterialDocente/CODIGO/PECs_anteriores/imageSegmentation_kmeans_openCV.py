'''
    IMAGE SEGMENTATION USING K-MEANS (UNSUPERVISED LEARNING)
    AUTHOR Paul Asselin

    command line arguments:
		python imageSegmentation.py K inputImageFilename outputImageFilename
	where K is greater than 2
'''
######################################################################################
#                        *********PARAMETROS CONFIGURAR **********
######################################################################################
inputName = "imagenes/estanteria.png"
K=10  # número de clases utilizadas para segmentar la imagen
iterations = 5 # número de iteraciones de kmeans


######################################################################################

import numpy as np
import sys
#from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import cv2
from timeit import default_timer as timer


#	Parse command-line arguments
#	sets K, inputName & outputName
if 0:

    if len(sys.argv) < 4:
    	print ("Error: Insufficient arguments, imageSegmentation takes three arguments")
    	sys.exit()
    else:
    	K = int(sys.argv[1])
    	if K < 3:
    		print("Error: K has to be greater than 2")
    		sys.exit()
    	inputName = sys.argv[2]
    	outputName = sys.argv[3]
        

#	Open input image
#image = Image.open(inputName)
img = cv2.imread(inputName)

image = img[::2,::2,:]
image = img


start=timer()

# reshape data to 1D BGR vector
Z = image.reshape((-1,3))
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
compactness,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# compactness : It is the sum of squared distance from each point to their corresponding centers.


# Now convert back into uint8, and make original image
center = np.uint8(center)
imageSeg = center[label.flatten()].reshape((image.shape))

end = timer()

print("Time elapsed: ", end - start) 

cv2.imshow("Orig",image)
cv2.waitKey(2)
cv2.imshow("Segmentation",imageSeg)
cv2.waitKey(0)

#  Time elapsed con bucles :  615.5471395189998
#  Time elapsed con openCV :  0.5526736390002043





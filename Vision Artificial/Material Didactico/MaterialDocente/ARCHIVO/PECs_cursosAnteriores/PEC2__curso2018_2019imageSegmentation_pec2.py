'''
    PEC2-2018-2019
'''
# PARAMETROS CONFIGuracion
MIN_AREA = 150   # area mínima blobs significativos
MIN_AREA = 250   # area mínima blobs significativos
K=10  # número de clases para segmentar la imagen
iterations = 5  # número de iteraciones en kmeans

inputName =  "/media/mrincon/SG5T_MRZ/MARI_TRA/_3_DOCENCIA/VisionArtificial-GRADO/__curso2018-2019/PEC2/PEC2_enunciado/estanteria.png"
outputName = "/media/mrincon/SG5T_MRZ/MARI_TRA/_3_DOCENCIA/VisionArtificial-GRADO/__curso2018-2019/PEC2/PEC2_enunciado/estanteriaSegmented.png"

import numpy as np
import sys
import matplotlib

from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

import cv2
from timeit import default_timer as timer
from matplotlib import pyplot as plt

#matplotlib.use('Qt5Agg', warn=False)


image = cv2.imread(inputName)

start=timer()


# reshape data to 1D BGR vector
Z = image.reshape((-1,3))
Z = np.float32(Z)

# clustering pixels:  define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
compactness,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# compactness : It is the sum of squared distance from each point to their corresponding centers.
# En center bbtenemos K valores, tantos como etiquetas hemos elegido  

if 0:
   # NOTA:     
    #otro operador que haace algo parecido a kmeans es meanshift, que devuelve para cada región el máximo local 
    # (dependerá de la configuración de parámetros las características de la región asociada a cada máximo local)
    # Con este operador se generan más etiquetas distintas (todos los máximos locales detectados) pero todavía puede 
    # haber varios blobs con la misma etiqueta (puede haber dos máximos locales exactamente iguales en toda la imagen)
    # Explicación: http://seiya-kumada.blogspot.com/2013/05/mean-shift-filtering-practice-by-opencv.html
    # USO: 
    imagenClusterizadaConMeanShift= cv2.pyrMeanShiftFiltering(img, 20, 55)

# Convert back into uint8, and make original image
center = np.uint8(center)
imageSeg = center[label.flatten()].reshape((image.shape))  #con formato imagen
imageSegV = imageSeg.reshape((-1,3)) # con formato columna

# En esta situación, los blobs se tocan y puede haber blobos distintos con el mismo valor 
# (se ha hecho una segmentación, pero no se han etiquetado los blobs). 
#
# El etiquetado de todos los blobs de la imagen se descompone por las etiquetas obtenidas con kmeans. 
# (los blobs que tengan la misma etiqueta están separados, si no pertenecerían al mismo blob)
# Etiquetado: Se eliminan los blobs pequeños y se ponen etiquetas correlativas
#   visualización de resultados intermedios solo a título ilustrativo
Iblob_acc = np.zeros(image.shape[0:2])
numLabels_acc = 0
for itc in center:
    print(itc)
    
    # Se seleccionan los pixeles de la etiqueta itc (3 canales)
    Vitc = (imageSegV[:,0]==itc[0]) & (imageSegV[:,1]==itc[1]) & (imageSegV[:,2]==itc[2]) 
    Blobs = np.uint8(Vitc.reshape((image.shape[0:2])))   # imagen binaria con los blobs de la clase colorBGR=itc

    
    # Se obtienen las características de los blobs
    # np.unique(a , row=True ) 
    # ret, labels = cv2.connectedComponents(imageSeg)
    connectivity = 8  # You need to choose 4 or 8 for connectivity type
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Blobs, connectivity , cv2.CV_32S)

    #plt.imshow(labels, cmap='gray')
    #plt.imshow(labels, cmap='Set1')
    if 1:
        plt.figure(1)
        #plt.imshow(labels, cmap='tab20')
        plt.imshow(labels, cmap='gray')
        plt.title("labels")
        plt.show()
        plt.pause(1)

    # se seleccionan solo los blobs con area mayor que MIN_AREA
    ind = np.array(np.where(stats[:, cv2.CC_STAT_AREA] > MIN_AREA) )   # indice lineal 1D    
    Ilsel = np.zeros(image.shape[0:2])
    for itind in ind.ravel():
        if itind:
            numLabels_acc +=1
            # indices de los pixeles del blob, poner los pixeles a cero
            indpixels = np.array(np.where(labels.ravel() == itind) )           
            Ilsel.ravel()[indpixels] = numLabels_acc   

    if 1:
        plt.figure(2)
        plt.imshow(Ilsel, cmap='tab20')
        plt.imshow(Ilsel, cmap='gray')
        plt.title("labels_FILT")
        plt.show()
        plt.pause(1)
        
    
    # Se añaden los pixeles de los blobs seleccionados a la imagen donde se acumulan todos los blobs    
    #Iblob_acc = Iblob_acc + (labels + numLabels_acc * (labels>0.5) )   
    #numLabels_acc = numLabels_acc + num_labels   
    Iblob_acc = Iblob_acc + Ilsel


    if 1:
        plt.figure(3)
        #plt.imshow(labels, cmap='tab20')
        plt.imshow(Iblob_acc, cmap='gray')
        plt.title("labels SEL")
        plt.show()
        plt.pause(1)
        kk=0
        
        
    #cv2.imshow("", labels)
    #cv2.waitKey(5)

# eliminar blobs pequeños y rellenarlos con el color de alrededor

# etiquetar blobs

# eliminar blobs pequeños


end = timer()

# para visualizar resultados de las etiquetas en el rango 0-255
Iblob_acc_Normalized = np.zeros(image.shape[0:2])
cv2.normalize( Iblob_acc, Iblob_acc_Normalized, 0, 255, cv2.NORM_MINMAX )
cv2.imshow("mainBlobs", np.uint8(Iblob_acc_Normalized) )
cv2.waitKey(2)

img_colorized = cv2.applyColorMap(np.uint8(Iblob_acc_Normalized) , cv2.COLORMAP_JET)
cv2.imshow("mainBlobs2", img_colorized )
cv2.waitKey(2)

print("Time elapsed: ", end - start) 
cv2.imshow("Orig",image)
cv2.waitKey(2)
cv2.imshow("Segmentation",imageSeg)
cv2.waitKey(2)

#  Time elapsed con bucles :  615.5471395189998
#  Time elapsed con openCV :  0.5526736390002043

# YA tenemos la imagen etiquetada, ahora habría que comprobar si cada blob cumple con los criterios 
# para considerarlo un blob correspondiente a una pila de objetos bien organizada o no.

# Las baldas se podrían identificar por su forma: son objetos orizontales alargados (ancho grande y alto pequeño) 
# Podemos distintuir los blobs que están justo encima de una balda:
# - bastaría con realizar una dilatación de los blobs "BALDA" con un elemento estructurante vertical y analizar la intersección con el resto de blobs
# elemento estructurante = [0 0 0 1 1 1 1] 
#
#
#
#
#
#






# Prueba para extraer las baldas:
# Observad que las baldas no son segmentables completamente por color. Además, en la zona izquierda, se añade una columna vertical que está conectada con la balda superior
# Podría ser necesario definir las regiones de las baldas directamente.
if 1:
    plt.figure(4)
    #plt.imshow(labels, cmap='tab20')
    plt.imshow(Iblob_acc, cmap='gray')
    plt.title("labels SEL")
    plt.show()           # en esta visualización se puede explorar el valor de los pixeles y sus coordenadas en la imagen
    plt.pause(1)
        
# OJO A ESTO !!!    
# tomamos de la imagen mostrada en la figura 4 los valores correspondientes a las baldas.
# Ojo, los algoritmos de segmentación y etiquetado no devuelven siempre las mismas etiquetas
# Para solucionarlo, podemos poner una lista de coordenadas que sepamos que pertenecen a las baldas y sacar de estas coordenadas el valor de las etiquetas 

## LblobsBaldas=[44, 45, 46, 47, 67]  cambiado por una asignación dinámica
CoordBaldas = [[112, 58], [142,244], [244, 228]]  
LblobsBaldas = np.zeros((size(CoordBaldas,0),1), np.uint)
for it in arange(size(CoordBaldas,0)):
   LblobsBaldas[it] =  Iblob_acc[CoordBaldas[it][1]][CoordBaldas[it][0]]

# imagen con las baldas
IblobsBaldas = np.zeros(image.shape[0:2])
for it in LblobsBaldas:
    print("it vale ",it)
    indpixels = np.array(np.where(Iblob_acc.ravel() == it) )           
    IblobsBaldas.ravel()[indpixels] = 1            
if 1:
    plt.figure(5)
    #plt.imshow(labels, cmap='tab20')
    plt.imshow(IblobsBaldas, cmap='gray')
    plt.title("labels SEL")
    plt.show()           
    plt.pause(1)


            
# Aplicamos la dilatación vertical para ver los blobs en contacto con las baldas por encima (ojo, valores decrecientes de y):            
kernel = np.zeros((5,5), np.uint8)
kernel[2,2]=kernel[3,2]=kernel[4,2]=1
IblobsBaldasDil = cv2.morphologyEx(IblobsBaldas, cv2.MORPH_DILATE, kernel)
if 1:
    plt.figure(5)
    #plt.imshow(labels, cmap='tab20')
    plt.imshow(IblobsBaldas+IblobsBaldasDil, cmap='gray')
    plt.title("labels SEL")
    plt.show()           
    plt.pause(1)

# ... y buscamos la intersección con los blobs
labelsContactoBaldas = IblobsBaldasDil * Iblob_acc
if 1:
    plt.figure(6)
    #plt.imshow(labels, cmap='tab20')
    plt.imshow(labelsContactoBaldas, cmap='gray')   # ver que se cogen algunas etiquetas por encima de las baldas. No todas porque hay ceros en medio => habría que mejorar el método 
    plt.title("labels SEL")
    plt.show()           
    plt.pause(1)

# También se podría analizar por debajo de la balda para eliminar los blobs correspondientes a huecos del análisis
    
# Ahora se podrían analizar las características de estos blobs para determinar si corresponden a pilas de objetos bien organizados o no.
# Los blobs que no están en contacto con las baldas son potenciales objetos mal apilados,...
# Lo ideal sería hacer un sistema experto para tratar con todas las reglas que definen el sistema
    









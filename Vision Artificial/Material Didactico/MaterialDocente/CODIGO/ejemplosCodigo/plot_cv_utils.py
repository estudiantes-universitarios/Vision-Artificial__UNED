import numpy as np 
import matplotlib.pyplot as plt
import cv2 

# Dibujar con matplotlib  una imagen leída con opencv
def plot_cv_img(input_image,is_gray=False):           
    # change color channels order for matplotlib 
    if not is_gray:
        plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    else:
        plt.imshow(input_image, cmap='gray', vmin=0, vmax=255)         

    # For easier view, turn off axis around image     
    plt.axis('off')  
    plt.show() 
    
##### 

# Dibujar con matplotlib  una imagen leída con opencv a tamaño real
def plot_cv_img_org(input_image, is_gray=False): 
    dpi = 80
    altura, anchura = input_image.shape[:2]
    figsize = anchura / dpi, altura / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    # change color channels order for matplotlib 
    if not is_gray:
        ax.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    else:
        ax.imshow(input_image, cmap='gray', vmin=0, vmax=255)     

#####

# Lista de subplot vertical u horizontal, en color o b/n
def subplot_cv_list(image_list, title_list, is_vertical = True, is_gray = False):     
    assert len(image_list) == len(title_list), "Listas de imágenes y títulos deben tener igual longitud"
    
    size = len(image_list)
    if(is_vertical):
        fig, ax = plt.subplots(nrows=size, ncols=1, figsize=(10,10))
    else:
        fig, ax = plt.subplots(nrows=1, ncols=size, figsize=(15,15))

    for i in range(0, size):
        if(not is_gray):
            ax[i].imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
        else:        
            ax[i].imshow(image_list[i], cmap='gray', vmin=0, vmax=255)          
        ax[i].set_title(title_list[i])
        ax[i].axis('off') 
    
    # plt.savefig('figures/03_convolution.png')

    plt.show()
    
#####

# Sub plots 2x2, en colores o b/n
def subplot_4x4(img1, img2, img3, img4, title1, title2, title3, title4, is_gray=False):   
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
    
    if(not is_gray):
        ax[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    else:        
        ax[0, 0].imshow(img1, cmap = 'gray', vmin=0, vmax=255)
    
    ax[0, 0].set_title(title1)
    ax[0, 0].axis('off')

    if(not is_gray):
        ax[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    else:        
        ax[0, 1].imshow(img2, cmap = 'gray', vmin=0, vmax=255)
    ax[0, 1].set_title(title2)
    ax[0, 1].axis('off') 

    if(not is_gray):
        ax[1, 0].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    else:        
        ax[1, 0].imshow(img3, cmap = 'gray', vmin=0, vmax=255) 
    ax[1, 0].set_title(title3)
    ax[1, 0].axis('off')

    if(not is_gray):
        ax[1, 1].imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    else:        
        ax[1, 1].imshow(img4, cmap = 'gray', vmin=0, vmax=255)
    ax[1, 1].set_title(title4)
    ax[1, 1].axis('off')

    plt.show()
    
#####    

# dibuja una imagen en b/n y sus histogramas simple y acumulado
def plot_image_histo(image, hist, cdf_norm, title):
    # para que el histograma tenga la misma altura que la imagen
    aspect = 256/cdf_norm.max()/1.6

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))

    ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)          
    ax[0].set_title(title)
    ax[0].axis('off')

    ax[1].plot(cdf_norm, color = 'b')
    ax[1].hist(image.flatten(),256,[0,256], color = 'r')
    ax[1].set_xlim([0,256])
    ax[1].set_aspect(aspect)
    ax[1].legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

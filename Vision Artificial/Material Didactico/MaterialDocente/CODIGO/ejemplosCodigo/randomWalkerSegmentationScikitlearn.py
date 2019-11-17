#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:24:50 2019
@author: mrincon

Adaptado de https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_random_walker_segmentation.html
Mirar también:
https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980

"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage

# Plot results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.4),
                                    sharex=True, sharey=True)



# Generate noisy synthetic data
data = skimage.img_as_float(binary_blobs(length=128, seed=1))

ax1.imshow(data, cmap='gray')
ax1.axis('off')
ax1.set_title('Blobs')


sigma = 0.35
data += np.random.normal(loc=0, scale=sigma, size=data.shape)
data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                         out_range=(-1, 1))

# The range of the binary image spans over (-1, 1).
# We choose the hottest and the coldest pixels as markers.
markers = np.zeros(data.shape, dtype=np.uint)
markers[data < -0.95] = 1
markers[data > 0.95] = 2

# Run random walker algorithm
labels = random_walker(data, markers, beta=10, mode='bf')

# Plot results
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
#                                    sharex=True, sharey=True)



ax2.imshow(data, cmap='gray')
ax2.axis('off')
ax2.set_title('Noisy data')
ax3.imshow(markers, cmap='magma')
ax3.axis('off')
ax3.set_title('Markers')
ax4.imshow(labels, cmap='gray')
ax4.axis('off')
ax4.set_title('Segmentation')

fig.tight_layout()
plt.show()
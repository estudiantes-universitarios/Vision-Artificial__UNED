#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:23:50 2019

Tomado de https://github.com/Permafacture/Py-Visvalingam-Whyatt

"""

from PyVisvalingamWhyatt.polysimplify import VWSimplifier

import numpy as np
from time import time
from matplotlib import pyplot as plt


n = 5000
thetas = np.linspace(0,2*np.pi,n)
pts1 = np.array([[np.sin(x),np.cos(x)] for x in thetas])
pts2 = np.array([[np.sin(x)+0.5,np.cos(x)+0.5] for x in thetas])

pts = np.vstack((pts1,pts2))

start=time()
simplifier = VWSimplifier(pts)
VWpts = simplifier.from_number(n/100)
end = time() 
print ("Visvalingam: reduced to %s points in %03f seconds" %(len(VWpts),end-start))
#50 points in .131 seconds on my computer


from rdp import rdp
start=time()
RDPpts = rdp(pts,epsilon=.00485) #found by trail and error
end = time()
print("Ramer-Douglas-Peucker: to %s points in %s seconds" %(len(RDPpts),end-start))
#40 points in 1.35 seconds on my computer

plt.plot(pts[:,0], pts[:,1],"b.-")
#plt.
plt.plot(VWpts[:,0], VWpts[:,1], "r.")

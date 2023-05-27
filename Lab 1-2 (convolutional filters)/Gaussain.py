# -*- coding: utf-8 -*-
"""
Created on Tue May 17 21:44:38 2022

@author: Nazmul
"""
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


path = "C:\\Users\\Nazmul\\Desktop\\cube.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

img_H, img_W = img.shape
kernel_size = 9
sigma = 1

mid = kernel_size//2
result = np.zeros(img.shape, dtype="uint8")
GKernel = np.zeros((kernel_size, kernel_size))


dividant = 2*pow(sigma, 2)
front_constant = 1/(2*math.pi*sigma*sigma)

for i in range(-mid, mid+1):
    for j in range(-mid, mid+1):
        GKernel[mid+i][mid+j] = math.exp(-(pow(i, 2)+pow(j, 2))/dividant)

GKernel = GKernel *  front_constant       
        
for x in range(mid, img_H-mid):
    for y in range(mid, img_W-mid):
        sum = 0
        for m in range(-mid, mid+1):
            for n in range(-mid, mid+1):
                sum += GKernel[m+mid, n+mid]*img[x-m, y-n]
        result[x, y] = int(sum)
        
        
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()       
    
        
        
        
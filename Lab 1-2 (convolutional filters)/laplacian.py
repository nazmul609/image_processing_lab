# -*- coding: utf-8 -*-
"""
Created on Wed May 18 00:48:31 2022

@author: Nazmul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


path = "H:/4.1/Image Processing and Computer Vision Lab (CSE 4128) 0.75/cube.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#img = cv2.resize(img, (256, 256))
kernel_size = 7
mid = kernel_size//2
sigma = 1
Kernel = np.zeros((kernel_size, kernel_size), dtype="float32")
result = np.zeros(img.shape, dtype="float32")


#gausss = cv2.GaussianBlur(img, (7, 7), 0)




gauss = np.zeros((7,7))
output = np.zeros((img.shape))
for i in range (-mid, mid+1):
    for j in range(-mid, mid+1):
        gauss[i+mid, j+mid] = np.exp( -(i**2+j**2)/(2*sigma**2) )
        
#plt.imshow(gauss)



for x in range(mid, img.shape[0]-mid):
    for y in range(mid, img.shape[1]-mid):
         sum = 0
         for u in range(-mid, mid+1):
             for v in range(-mid, mid+1):
                 sum += gauss[u+mid, v+mid] * img[x-u, y-v]
                 
         output[x, y] = sum
         
plt.imshow(output, 'gray')
plt.show()

#laplace
def laplace_filter(x, y, sigma):
    weight = -1/(math.pi*pow(sigma, 4))
    weight2 = (x*x + y*y)/(2*sigma*sigma)

    return weight*(1-weight2)*math.exp(-weight2)



for i in range(-mid, mid+1):
    for j in range(-mid, mid+1):
        Kernel[mid+i, mid+j] = laplace_filter(i, j, sigma)

for x in range(mid, img.shape[0]-mid):
    for y in range(mid, img.shape[1]-mid):
        sum = 0
        for m in range(-mid, mid+1):
            for n in range(-mid, mid+1):
                sum += Kernel[m+mid, n+mid]*output[x-m, y-n]
        result[x, y] = sum

plt.imshow(result, 'gray')
plt.show()
        
op2 = np.zeros((img.shape[0],img.shape[1]),np.float32)
for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        op2[i][j] = img[i][j] + result[i][j]    
        


#bilateral




plt.imshow(op2, 'gray')
plt.show()

plt.imshow(img, cmap='gray')
plt.show()
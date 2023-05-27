# -*- coding: utf-8 -*-
"""
Created on Wed May 18 00:47:52 2022

@author: Nazmul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt



path = "C:\\Users\\Nazmul\\Desktop\\cube.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_H, img_W = img.shape
kernel_size = 5
kernel_len = kernel_size*kernel_size
mid = kernel_size // 2

result = np.zeros(img.shape, dtype="uint8")

def filter_mean(x, y):
    sum = 0
    for i in range(-mid, mid+1):
        for j in range(-mid, mid + 1):
            sum += img[x+i, y+j]
    return int(sum/kernel_len)


for i in range(mid, img_H - mid):
    for j in range(mid, img_W - mid):
        result[i, j] = filter_mean(i, j)

    
plt.imshow(result, 'gray')

plt.show()


import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity


path = "C:\\Users\\Nazmul\\Desktop\\cube.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (200, 200))
img_H = img.shape[0]
img_W = img.shape[1]


kernel_size = 5
sigma = 1
rsigma = 5

out = np.empty((img_H, img_W), dtype="float32")
mid = math.floor( kernel_size / 2 )

img = cv2.copyMakeBorder(img, mid, mid, mid, mid, cv2.BORDER_REPLICATE)

Gaussian_Kernel = np.zeros((kernel_size, kernel_size))


Gaussian_Kernel = np.array(([1,2,4,2,1],
                   [2,4,8,4,1],
                   [4,8,16,8,4],
                   [2,4,8,4,2],
                   [1,2,4,2,1]), np.float32)

def Gaussian_Intensity_Kernel(x, y):
    
    intensityKernel = np.zeros((kernel_size, kernel_size))
    dividant1 = 2 * pow(rsigma, 2)

    for i in range(-mid, mid+1):
        for j in range(-mid, mid+1):
            val = math.exp(-(pow(int(img[x, y]) - int(img[x+i, y+j]), 2)) / dividant1) # range domain
            intensityKernel[mid + i][mid + j] = val

    GI_Kernel = np.multiply(Gaussian_Kernel, intensityKernel)

    return GI_Kernel


def bilateralFilter():
    result = np.zeros((img_H, img_W), dtype=np.float32)
    n = math.floor(kernel_size/2)
    for x in range(img_H):
      for y in range(img_W):
        sum = 0
        
        kernel = Gaussian_Intensity_Kernel(x, y)
        kernel_sum = kernel.sum()

        for i in range(kernel_size):
          for j in range(kernel_size):
            sum += kernel[i, j]*img[x-i-n, y-j-n]
        result[x, y] = sum/kernel_sum
    
    result = rescale_intensity(result, in_range=(0, 255))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()



bilateralFilter()


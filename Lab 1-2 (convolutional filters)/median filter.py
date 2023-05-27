import numpy as np
import cv2
import matplotlib.pyplot as plt


path = "C:\\Users\\Nazmul\\Desktop\\cube.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


img_H, img_W = img.shape
kernel_size = 7
kernel_len = kernel_size*kernel_size
mid = kernel_size//2

result = np.zeros(img.shape, dtype="uint8")

def filter_median(x,y):
    pixel_value = []
    for i in range (-mid, mid+1):
        for j in range(-mid, mid+1):
            pixel_value.append(img[x-i, y-j])
    pixel_value.sort()
    return pixel_value[kernel_len//2]

for i in range(mid, img_H-mid):
    for j in range(mid, img_W - mid):
        result[i, j] = filter_median(i, j)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()       
    

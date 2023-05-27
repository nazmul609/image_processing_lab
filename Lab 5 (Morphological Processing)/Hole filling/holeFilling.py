#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import cv2 
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


path = "H:/4.1/Image Processing and Computer Vision Lab (CSE 4128) 0.75/lab work 5/th_img2.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, "gray")


# In[ ]:





# In[3]:


xp = []
yp = []

def eventClick(event, x, y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        xp.append(y)
        yp.append(x)


# In[4]:


cv2.imshow("img", img)

cv2.setMouseCallback("img", eventClick)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


t, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
img=img//255


# In[6]:


X = np.zeros((img.shape), np.uint8)
se = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
for i in range(len(xp)):
    X[xp[i],yp[i]]=1
Ac = 1 - img
while(True):
    prevX = X
    a = cv2.dilate(X, se)
    X = np.bitwise_and(a,Ac)
    plt.imshow(X,"gray")
    plt.show()
    if np.array_equal(X,prevX):
        break


# In[7]:


plt.imshow(X)


# In[8]:



X = np.bitwise_or(X, img)
plt.imshow(X,"gray")


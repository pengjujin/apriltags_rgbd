#!/usr/bin/python

import cv2
import numpy as np 
from matplotlib import pyplot as plt

img_depth = cv2.imread('../Data/rawData/dpt0.jpg')
img_rgb = cv2.imread('../Data/rawData/rgb0.jpg')



plt.subplot(121)
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)), plt.title('color')
plt.subplot(122)
plt.imshow(img_depth), plt.title('depth')
plt.axis("off")
plt.show()

img_rgb_tag = img_rgb[288.5:314.5, 459.5:490.5]
img_depth_tag = img_depth[288.5:314.5, 459.5:490.5]

plt.figure
plt.subplot(121)
plt.imshow(img_rgb_tag), plt.title('tag color')
plt.subplot(122)
plt.imshow(img_depth_tag), plt.title('img_depth_tag')
plt.axis("off")
plt.show()

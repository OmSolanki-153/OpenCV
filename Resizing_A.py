import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('C://Practicals//OpenCV//Batman Arkham Knight.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, channels = img_rgb.shape

resize_img = cv2.resize(img_rgb,(0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

plt.subplot(121), plt.imshow(img_rgb), plt.title('Original Image')
plt.subplot(122), plt.imshow(resize_img), plt.title('Zoomed Image')
plt.show()

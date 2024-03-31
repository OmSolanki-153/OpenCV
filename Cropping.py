import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('D://Part1Msc//CV//GameBoyy.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, channels = img_rgb.shape


#We choose to get 200 pixels from 100 to 300 on both axes.
dst = img[100:300, 100:300]

plt.subplot(121),plt.imshow(img_rgb), plt.title("Original Image")
plt.subplot(122),plt.imshow(dst), plt.title("Output Image")
plt.show()

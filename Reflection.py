import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('C://Practicals//OpenCV//Batman Arkham Knight.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, channels = img_rgb.shape

#To flip horizontally
M = np.float32([[1, 0, 0], [0, -1, rows],[0,  0, 1]])

#To flip vertically
#M = np.float32([[-1,0,cols],[0,1,0],[0,0,1]])

dst = cv2.warpPerspective(img_rgb,M,(cols,rows))

plt.subplot(121),plt.imshow(img_rgb), plt.title("Original Image")
plt.subplot(122),plt.imshow(dst), plt.title("Output Image")
plt.show()

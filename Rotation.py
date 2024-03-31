import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('C://Practicals//OpenCV//Batman Arkham Knight.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, channels = img_rgb.shape

center = (cols // 2, rows // 2)
angle = -90
scale = 1

rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

rotated_image = cv2.warpAffine(img_rgb, rotation_matrix, (cols,rows))

fig, axs = plt.subplots(1,2,figsize=(7, 4))

axs[0].imshow(img_rgb)
axs[0].set_title("Original image")

axs[1].imshow(rotated_image)
axs[1].set_title("Rotated image")

plt.tight_layout()
plt.show()

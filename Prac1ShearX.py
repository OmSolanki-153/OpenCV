import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('D://Part1Msc//CV//GameBoyy.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, channels = img_rgb.shape

M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])

dst = cv2.warpPerspective(img_rgb, M,
                                 (int(cols*1.5),
                                 int(rows*1.5)))

plt.subplot(121),plt.imshow(img_rgb), plt.title("Original Image")
plt.subplot(122),plt.imshow(dst), plt.title("Output Image")
plt.show()

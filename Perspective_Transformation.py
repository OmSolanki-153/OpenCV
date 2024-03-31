import cv2
import matplotlib.pyplot as plt
import numpy as np

#This program will act like camscanner the only difference is your have to provide
#co-ordinate so the image can be cropped properly.



# As input take something like paper on table with 4 corners.
img = cv2.imread('C://Practicals//OpenCV//Batman Arkham Knight.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, channels = img_rgb.shape

#Add the 4 corners value in pts1
pts1 = np.float32([[100,51],[260,37],[110,325], [250,320]])

pts2 = np.float32([[0,0],[600,0],[0,600], [600,600]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img_rgb,M,(600,600))

#While executing first time comment the line below to obtain original Image
#Then hover your mouse pointer at the edge of the table.
plt.subplot(121),plt.imshow(img_rgb), plt.title("Original Image")
plt.subplot(122),plt.imshow(dst), plt.title("Scanned Image")
plt.show()



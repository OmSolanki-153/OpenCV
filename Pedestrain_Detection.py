import cv2
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


image = cv2.imread('pedstrian3.jpg')

image = imutils.resize(image,
                       width=min(400,image.shape[1]))

(regions, _) = hog.detectMultiScale(image,
                                    winStride=(4,4),
                                    padding=(4,4),
                                    scale=1.05)

for (x, y, w, h) in regions:
    cv2.rectangle(image,(x,y),
                  (x + w, y + h),
                  (0, 0, 255),2)

cv2.imshow("Image",image)

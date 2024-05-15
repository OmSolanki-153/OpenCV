import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray,(7,6), None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners,(11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()
ret, mtx, dist, rvec, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("Camera matrix: ")
print(mtx)
print("dist: ")
print(dist)
print("rvecs: ")
print(rvec)
print("tvecs: ")
print(tvecs)

img=cv.imread('left08.jpg')
h,w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


dst = cv.undistort(img,mtx,dist,None, newcameramtx)

x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png',dst)

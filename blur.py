import cv2
import numpy as np

img = cv2.imread('img2.jpg')
img = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
cv2.imshow('Original', img)

size = 100

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[: , int((size-1)/2)] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# applying the kernel to the input image
output = cv2.filter2D(img, -1, kernel_motion_blur)

cv2.imshow('Motion Blur', output)
cv2.waitKey(0)

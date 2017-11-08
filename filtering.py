import numpy as np
import cv2

img = cv2.imread('img.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Original',img)
noise = np.random.normal(0,15,img.shape).reshape(img.shape)
noisy = (img + noise).astype('uint8')
# cv2.imshow('Noisy',noisy)
# cv2.imwrite('noisy2.jpg',noisy)
denoised = cv2.fastNlMeansDenoisingColored(noisy,None,5,5,5,21)
den2 = cv2.medianBlur(denoised,5)
cv2.imshow('Denoised',np.hstack((noisy,denoised,den2)))
cv2.waitKey(0)
cv2.destroyAllWindows()

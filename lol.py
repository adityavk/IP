import numpy as np
import cv2

image=cv2.imread('faces/RED/img12.jpg')
cv2.imshow('new',image)

img2 = image.copy().astype(int)

roi = img2[142:178,308:367]
b,g,r = cv2.split(roi)

redness = np.power(r,2)/(np.power(b,2) + np.power(g,2) + 1.0)

np.savetxt('redness.txt',redness,fmt="%d")

cv2.waitKey(0)
cv2.destroyAllWindows()

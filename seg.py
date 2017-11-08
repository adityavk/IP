import numpy as np
import cv2

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, image, clone

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
		image = clone.copy()

	elif cropping and event == cv2.EVENT_MOUSEMOVE:
		image = clone.copy()
		cv2.rectangle(image, refPt[0], (x,y), (0, 255, 0), 2)
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

image = cv2.imread('img4.jpg')
clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)
# # keep looping until the 'q' key is pressed
# while True:
#     # display the image and wait for a keypress
#     cv2.imshow("image", image)
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the 'r' key is pressed, reset the cropping region
#     if key == ord("r"):
#         image = clone.copy()
#
#     # if the 'c' key is pressed, break from the loop
#     elif key == ord("c"):
#         break
#
# cv2.destroyAllWindows()

img=clone.copy()
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (1,1,img.shape[1],img.shape[0])
print(rect)
# rect = (315,12,657,360)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imshow('Final',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

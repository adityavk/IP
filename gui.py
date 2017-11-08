from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import numpy as np
import cv2

refPt = []
cropping = False
name=''

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



def callback():
    global name
    name= askopenfilename()
    canvas2.image=''
    if len(name)>0:
        im = Image.open(name)
        im = im.resize((canvas_width,canvas_height))
        img = ImageTk.PhotoImage(im)
        canvas.image = img
        canvas.create_image(0, 0, anchor=NW, image=img)

def blur():
    global image, clone
    image = cv2.imread(name)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:
    	# display the image and wait for a keypress
    	cv2.imshow("image", image)
    	key = cv2.waitKey(1) & 0xFF

    	# if the 'r' key is pressed, reset the cropping region
    	if key == ord("r"):
    		image = clone.copy()

    	# if the 'c' key is pressed, break from the loop
    	elif key == ord("c"):
    		break

    cv2.destroyAllWindows()

    size = 20

    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2) , :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    output = cv2.filter2D(clone.copy(), -1, kernel_motion_blur)
    output[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

    im2 = cv2.resize(output,(canvas_width,canvas_height))
    im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    im2=Image.fromarray(im2)
    img2 = ImageTk.PhotoImage(im2)
    canvas2.image = img2
    canvas2.create_image(0, 0, anchor=NW, image=img2)




canvas_width = 512
canvas_height = 512

master = Tk()

canvas = Canvas(master, width=canvas_width, height=canvas_height)
canvas.config(bd=5)
canvas.pack(side = 'left', fill=Y, expand = True)

canvas2 = Canvas(master, width=canvas_width, height=canvas_height)
canvas2.pack(side = 'right', fill=Y, expand = True)


errmsg = 'Error!'
Button(text='Open Image', command=callback).pack(fill=X)
Button(text='Selective Blur', command=blur).pack(fill=X)
mainloop()

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import numpy as np
import cv2

import cleaned_eye

refPt = []
cropping = False
name=''

# def click_and_crop(event, x, y, flags, param):
# 	# grab references to the global variables
# 	global refPt, cropping, image, clone
#
# 	# if the left mouse button was clicked, record the starting
# 	# (x, y) coordinates and indicate that cropping is being
# 	# performed
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		refPt = [(x, y)]
# 		cropping = True
# 		image = clone.copy()
#
# 	elif cropping and event == cv2.EVENT_MOUSEMOVE:
# 		image = clone.copy()
# 		cv2.rectangle(image, refPt[0], (x,y), (0, 255, 0), 2)
# 	# check to see if the left mouse button was released
# 	elif event == cv2.EVENT_LBUTTONUP:
# 		# record the ending (x, y) coordinates and indicate that
# 		# the cropping operation is finished
# 		refPt.append((x, y))
# 		cropping = False
#
# 		# draw a rectangle around the region of interest
# 		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
# 		cv2.imshow("image", image)

def resize_dimensions(shape):
	# print(shape)
	w,h = shape[:2]
	if w >= h:
		return (canvas_width,int((canvas_width*h)/(1.0*w)))
	else:
		return (int((canvas_height*w)/(1.0*h)),canvas_height)

def callback():
	global name
	name= askopenfilename()
	canvas2.image=''
	if len(name)>0:
		im = Image.open(name)
		im = im.resize(resize_dimensions(im.size))
		img = ImageTk.PhotoImage(im)
		canvas.image = img
		canvas.create_image(canvas_width/2, canvas_width/2, anchor=CENTER, image=img)

def output(image):
	img = cv2.resize(image,resize_dimensions(image.shape))
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	canvas2.image = img
	canvas2.create_image(canvas_width/2, canvas_width/2, anchor=CENTER, image=canvas2.image)

def redeye():
	img = cv2.imread(name)
	img = cv2.resize(img,resize_dimensions(img.shape))
	output(cleaned_eye.main(img))

# def main():
canvas_width = 512
canvas_height = 512

master = Tk()
master.attributes("-zoomed",True)
canvas = Canvas(master, width=canvas_width, height=canvas_height,
	highlightbackground='black', highlightcolor='black', highlightthickness=1)
# canvas.config(bd=5)
# canvas.pack(side = 'left', fill=Y, expand = True)
canvas.grid(row=2,column=1,rowspan=2,padx=50,pady=50)

canvas2 = Canvas(master, width=canvas_width, height=canvas_height,
	highlightbackground='black', highlightcolor='black', highlightthickness=1)
# canvas2.pack(side = 'right', fill=Y, expand = True)
canvas2.grid(row=2,column=5,rowspan=2,padx=50,pady=50)

errmsg = 'Error!'
Button(text='Open Image', command=callback,
	highlightbackground='black', highlightcolor='black', highlightthickness=1).grid(row=6,column=3)
# Button(text='Display Output', command=output,
# 	highlightbackground='black', highlightcolor='black', highlightthickness=1).grid(row=6,column=5)
Button(text='Red Eye Correction', command=redeye,
	highlightbackground='black', highlightcolor='black', highlightthickness=1).grid(row=2,column=3)
Button(text='Image inpainting', command=redeye,
	highlightbackground='black', highlightcolor='black', highlightthickness=1).grid(row=3,column=3)
mainloop()
#
# if __name__=="__main__":
# 	main()

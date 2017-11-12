from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import numpy as np
import cv2
import inpainting
import red_eye

name=''

def resize_dimensions(shape):
	h,w = shape[:2]
	if h >= w:
		return (int((canvas_width*w)/(1.0*h)),canvas_width)
	else:
		return (canvas_height,int((canvas_height*h)/(1.0*w)))

def callback():
	global name
	name= askopenfilename()
	canvas2.image=''
	if len(name)>0:
		im = Image.open(name)
		im = im.resize(resize_dimensions(im.size)[::-1])
		img = ImageTk.PhotoImage(im)
		canvas.image = img
		canvas.create_image(canvas_width/2, canvas_width/2, anchor=CENTER, image=img)

def output(image):
	if image.shape[0]>canvas_height or image.shape[1]>canvas_width:
		img = cv2.resize(image,resize_dimensions(image.shape))
	else:
		img = image.copy()
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	canvas2.image = img
	canvas2.create_image(canvas_width/2, canvas_width/2, anchor=CENTER, image=canvas2.image)

def clear():
	canvas2.delete('all')

def redeye():
	# clear()
	img = cv2.imread(name)
	if img is None:
		return
	img = cv2.resize(img,resize_dimensions(img.shape))
	output(red_eye.main(img))

def inpaint():
	global name
	img = cv2.imread(name)
	if img is None:
		return
	inpainted = inpainting.main(img.copy(),4)
	inpainted = cv2.resize(inpainted,resize_dimensions(img.shape),interpolation = cv2.INTER_CUBIC)
	output(inpainted)

canvas_width = 512
canvas_height = 512
master = Tk()
master.title('Photo Editor')
master.attributes("-zoomed",True)

canvas = Canvas(master, width=canvas_width, height=canvas_height,
	highlightbackground='black', highlightcolor='black', highlightthickness=1)
canvas.grid(row=2,column=1,rowspan=2,padx=50,pady=50)

canvas2 = Canvas(master, width=canvas_width, height=canvas_height,
	highlightbackground='black', highlightcolor='black', highlightthickness=1)
canvas2.grid(row=2,column=5,rowspan=2,padx=50,pady=50)

Button(text='Open Image', command=callback,
	highlightbackground='black', highlightcolor='black', highlightthickness=1).grid(row=6,column=3)

Button(text='Red Eye Correction', command=redeye,
	highlightbackground='black', highlightcolor='black', highlightthickness=1).grid(row=2,column=3)

Button(text='Image inpainting', command=inpaint,
	highlightbackground='black', highlightcolor='black', highlightthickness=1).grid(row=3,column=3)

mainloop()

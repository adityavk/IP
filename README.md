# PhotoEditor
A GUI based application developed as Assignment 2 of the course EE604A: Digital Image Processing (IITK). This is an attempt to automate red-eye correction and object removal. Red-eye correction is a standard solved problem, and the implementation combines various state-of-the-art techniques used in image processing like thresholding, image growing, flood filling, closing, etc. The object removal part is an implementation of Exemplar-based inpainting [algorithm](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1323101) proposed by Criminisi et.al.

# Getting Started
Following Python3 packages will be required to run the application:
* tkinter
* cv2
* PIL
* matplotlib
* numpy

# Deployment
To run the application just run the command:
```
python integrated.py
```

# Author
Aditya Vikram

# Acknowledgements
* @veslam

import numpy as np
import cv2
import sys
import os

def Redness(b,g,r):
    return np.power(r,2)/(np.power(b,2) + np.power(g,2) + 1.0)

def neighbor(point, width, height):
    x, y = point
    left = max(x-1,0)
    right = min(x+1,width-1)+1
    top = max(y-1,0)
    bottom = min(y+1,height-1)+1
    return [(i,j) for i in np.arange(left,right) for j in np.arange(top,bottom)]

def region_growing(image,mask,weak_threshold):
    x,y = np.where(mask)
    points = list(zip(x,y))
    processed=[]
    while len(points):
        point = points[0]
        for (x,y) in neighbor(point, image.shape[1], image.shape[0]):
            if (x,y) not in processed:
                red = Redness(image[x,y,0], image[x,y,1], image[x,y,2])
                # print((x,y),red)
                processed.append((x,y))
                if  red> weak_threshold:
                    mask[x,y]=255
                    points.append((x,y))
        points.pop(0)
    return mask

def fillHoles(binary):
    # Mask used to flood filling
    h, w = binary.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    clone = binary.copy()

    # Floodfill from point (0, 0)
    cv2.floodFill(clone, mask, (0,0), 255);

    # Invert floodfilled image
    invert = cv2.bitwise_not(clone)

    # Combine the two images to get the foreground.
    return  binary | invert

def main(img):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.03,minNeighbors=7)

    strong_threshold = 4
    weak_threshold = 2

    for (ex,ey,ew,eh) in eyes:
        img2 = img.copy().astype(int)
        roi = img2[ey:ey+ew,ex:ex+ew]
        b,g,r = cv2.split(roi)

        redness = np.power(r,2)/(np.power(b,2) + np.power(g,2) + 1.0)

        binary = (255 * (redness > strong_threshold)).astype(np.uint8)

        ratio = 0.03

        kernel1_size = int(round(1.0*ratio*ew))
        kernel_size = int(round(2.0*ratio*ew))

        if kernel_size<2:
            kernel1_size = 2
            kernel_size = 4

        output = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

        if output[0] > 1:
            ind = np.argmax(output[2][1:, cv2.CC_STAT_AREA])
            w,h = output[2][ind+1,cv2.CC_STAT_WIDTH],output[2][ind+1,cv2.CC_STAT_HEIGHT]
            kernel1_size = min(kernel1_size, int(w/2.0), int(h/2.0))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))

        if kernel1_size > 0:
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel1_size,kernel1_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)

        binary = region_growing(roi,binary,weak_threshold)

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        binary = fillHoles(binary)

        output = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
        for i in range(1,output[0]):
            width_ratio = output[2][i,cv2.CC_STAT_WIDTH]/(1.0*ew)
            height_ratio = output[2][i,cv2.CC_STAT_HEIGHT]/(1.0*ew)
            area_ratio = 2.0*output[2][i,cv2.CC_STAT_AREA]/(output[2][i,cv2.CC_STAT_HEIGHT]*output[2][i,cv2.CC_STAT_WIDTH])
            if width_ratio < 0.1 or width_ratio > 0.5:
                binary[output[1] == i] = 0
            elif height_ratio < 0.1 or height_ratio > 0.5:
                binary[output[1] == i] = 0
            elif output[2][i,cv2.CC_STAT_WIDTH] > 2*output[2][i,cv2.CC_STAT_HEIGHT] or output[2][i,cv2.CC_STAT_HEIGHT] > 2*output[2][i,cv2.CC_STAT_WIDTH]:
                binary[output[1] == i] = 0
            elif area_ratio < 0.9:
                binary[output[1] == i] = 0

        mean = ((g[binary==255].astype(int)+b[binary==255].astype(int))/2)
        b[binary==255]=mean
        g[binary==255]=mean
        r[binary==255]=mean

        new = cv2.merge((b,g,r))

        img[ey:ey+ew,ex:ex+ew] = new.astype(np.uint8)
    return img
    
if __name__ == "__main__":
    main(sys.argv[1])

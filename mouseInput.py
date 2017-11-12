import cv2
import numpy as np
import sys

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,img,image

    cv2.imshow('image',image)

    brushwidth = 1

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(image,(x,y),brushwidth,255,-1)
            cv2.circle(img,(x,y),brushwidth,255,-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(image,(x,y),brushwidth,255,-1)
        cv2.circle(img,(x,y),brushwidth,255,-1)

def main(input_image):
    global ix,iy,drawing,img,image
    drawing = False # true if mouse is pressed
    ix,iy = -1,-1
    image = input_image.copy()
    img = np.zeros(image.shape[:2], np.uint8)

    clone1=image.copy()
    clone2=img.copy()

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        # img = np.zeros((512,512,3), np.uint8)
        cv2.imshow('image',image)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('r'):
            image=clone1.copy()
            img=clone2.copy()
        elif k == 13:
            break

    cv2.destroyAllWindows()

    # cv2.imshow('new',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('new',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # seed_pt =
    connectivity = 8
    flooded = closed.copy()
    # cv2.floodFill(flooded, mask, (w-1,h-1), (255,255,255))
    # cv2.imshow('new',flooded)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.floodFill(flooded, mask, (0,0), (255,255,255))
    # cv2.imshow('new',flooded)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    invert = 255-flooded
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    opened = cv2.morphologyEx(invert, cv2.MORPH_OPEN, kernel)
    withBoundary = opened | img
    final = cv2.morphologyEx(withBoundary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('report/mask2.jpg',final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return final

if __name__ =="__main__":
    imagee=cv2.imread(sys.argv[1])
    main(imagee)

import sys
import numpy as np
import cv2
import mouseInput
import matplotlib.pyplot as plt
import ctypes
import time

def gradients(source):
    global img, grad_x, grad_y
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)/255.0
    grad_x[source==0] = 0.0
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)/255.0
    grad_y[source==0] = 0.0
    return grad_x, grad_y

def initialFillFront(target):
    maskBoundary = cv2.Laplacian(target.astype(np.float32),cv2.CV_32F)#,LAPLACIAN_KERNEL)
    return np.argwhere(maskBoundary>0)

def normals(source, fillFront):
    gradx_kernel = np.zeros((3, 3), dtype = float)
    gradx_kernel[1, 0] = -1.0
    gradx_kernel[1, 2] = 1.0
    source_float = source.astype(np.float32)
    normal_y = (-1.0 * cv2.filter2D(source_float, cv2.CV_32F, gradx_kernel))[fillFront[:,0],fillFront[:,1]]
    normal_x = cv2.filter2D(source_float, cv2.CV_32F, gradx_kernel.T)[fillFront[:,0],fillFront[:,1]]
    magnitude = np.sqrt(normal_x**2 + normal_y**2)
    magnitude[magnitude == 0.0] = 1.0
    normal_x /= magnitude
    normal_y /= magnitude
    return normal_x,normal_y


def patchSides(point, sizeY, sizeX):
    global halfsize
    y, x = point
    left = max(0, (x - halfsize))
    right = min((sizeX-1), (x + halfsize))
    top = max(0, (y - halfsize))
    bottom = min((sizeY-1), (y + halfsize))
    return left, right, top, bottom

def patchConfidence(point, confidence, source):
    left, right, top, bottom = patchSides(point, confidence.shape[0], confidence.shape[1])
    patch = confidence[top:(bottom+1), left:(right+1)]
    sourcePatch = source[top:(bottom+1), left:(right+1)]
    return (patch[sourcePatch==1]).sum()/(1.0*patch.size)

def updateConfidence(confidence, fillFront, source):
    for point in fillFront:
        confidence[point[0],point[1]] = patchConfidence((point[0],point[1]),confidence,source)
    return confidence

def updateData(source,fillFront):
    global grad_x, grad_y
    normals_x, normals_y = normals(source,fillFront)
    return np.fabs(normals_x*grad_x[fillFront[:,0],fillFront[:,1]] + normals_y*grad_y[fillFront[:,0],fillFront[:,1]])

def maxPriorityPatch(fillFront, source, confidence):
    priority = updateData(source,fillFront) * (updateConfidence(confidence,fillFront,source)[fillFront[:,0],fillFront[:,1]])
    return fillFront[np.argmax(priority),:]#, confidence

def initPatchlist():
    global globalPatches, globalLocations, img, halfsize
    height, width = img.shape[:2]
    globalPatches=[]
    globalLocations=[]
    size = 2*halfsize+1
    for row in range(height - size):
        for column in range(width - size):
            if np.count_nonzero(source_copy[row:(row+size), column:(column+size)]) == (size*size):
                globalPatches.append(img[row:(row+size), column:(column+size)])
                globalLocations.append((row,column))
    globalPatches = np.array(globalPatches)
    globalLocations = np.array(globalLocations)

def bestPatch(targetPatchY, targetPatchX, source, confidence, target, iteration):
    global img, halfsize, grad_x, grad_y, globalPatches, globalLocations

    height, width = img.shape[:2]
    left, right, top, bottom = patchSides((targetPatchY,targetPatchX), height, width)
    patchX = right - left + 1
    patchY = bottom - top + 1

    if patchY==0 or patchX==0:
        return

    targetPatch = img[top:(bottom+1), left:(right+1)].astype(np.float32)
    if patchY == (2*halfsize+1) and patchX == (2*halfsize+1):
        possiblePatches = globalPatches
        locations = globalLocations
    else:
        possiblePatches = []
        locations=[]
        for row in range(height - patchY):
            for column in range(width - patchX):
                if np.count_nonzero(source_copy[row:(row+patchY), column:(column+patchX)]) == (patchX*patchY):
                    possiblePatches.append(img[row:(row+patchY), column:(column+patchX)])
                    locations.append((row,column))
        possiblePatches = np.array(possiblePatches)
        locations = np.array(locations)

    inSource = source[top:(bottom+1), left:(right+1)]

    error = (((possiblePatches - targetPatch)**2)*np.dstack([inSource]*3)).sum(axis=(1,2,3))
    minimas = np.argwhere(error == error.min())
    minimas = minimas.reshape(minimas.shape[0],)
    variances = np.var(possiblePatches[minimas,:,:,:],axis=(1,2,3))
    ind = minimas[np.argmin(variances)]
    bestPatch = possiblePatches[ind].astype(np.uint8)
    bestY, bestX = locations[ind]

    img[top:(bottom+1), left:(right+1)][~inSource] = bestPatch[~inSource]

    confidence[top:(bottom+1), left:(right+1)][~inSource] = confidence[targetPatchY,targetPatchX]

    grad_x[top:(bottom+1), left:(right+1)] = grad_x[bestY:(bestY+patchY), bestX:(bestX+patchX)]
    grad_y[top:(bottom+1), left:(right+1)] = grad_y[bestY:(bestY+patchY), bestX:(bestX+patchX)]

    target[top:(bottom+1), left:(right+1)][~inSource] = 0
    source[top:(bottom+1), left:(right+1)][~inSource] = 1


def main(image, size):
    global halfsize, img, mask, grad_x, grad_y, source_copy
    original_mask = mouseInput.main(image)

    if (image.shape[0]*image.shape[1])>160000:
        if image.shape[0] <= image.shape[1]:
            image = cv2.resize(image,(400,int((image.shape[0]*400)/image.shape[1])),interpolation = cv2.INTER_CUBIC)
            original_mask = cv2.resize(original_mask,(400,int((image.shape[0]*400)/image.shape[1])),interpolation = cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image,(int((image.shape[1]*400)/image.shape[0]),400),interpolation = cv2.INTER_CUBIC)
            original_mask = cv2.resize(original_mask,(400,int((image.shape[0]*400)/image.shape[1])),interpolation = cv2.INTER_CUBIC)
    img = image.copy()

    halfsize = size

    mask = original_mask.copy()

    mask[mask<10]=0

    confidence = np.ones_like(mask)
    confidence[mask > 10] = 0

    source = confidence.astype(bool)
    source_copy = source.copy()

    target = ~source

    confidence = confidence.astype(float)

    gradients(source)
    fillFront = initialFillFront(target)
    initPatchlist()
    iterations = 0

    while fillFront.shape[0]>0:
        print(iterations)
        iterations += 1
        y, x = maxPriorityPatch(fillFront, source, confidence)
        bestPatch(y, x, source, confidence, target, iterations)
        fillFront = initialFillFront(target)
    return img

if __name__ == "__main__":
    main(cv2.imread(sys.argv[1]), int(sys.argv[2]))

import ctypes
import numpy as np
import cv2
# print ctypes.windll.library.square(4) # windows
# c_float_p = ctypes.POINTER(ctypes.c_float)
# data = numpy.array([0.1, 0.1,0.2, 0.2])
# data = data.astype(numpy.float32)
# data_p = data.ctypes.data_as(c_float_p)
# print(ctypes.CDLL('library.so').square())#data_p,4)

# img = cv2.imread('tests/image2.jpg')

indata = np.arange(1215, dtype=np.double).reshape(5,9,9,3)
outdata = indata[2,:,:,:].copy()#np.random.rand(9,9,3)#.astype(np.double)
# print(indata.shape[0],indata.shape[1],indata.shape[2],indata.shape[3])
index = np.zeros((1,),dtype=np.int64)
mask = np.arange(81).reshape(9,9)
mask = (mask%2).astype(np.bool)
lib = ctypes.cdll.LoadLibrary('./library.so')
fun = lib.cfun
# Here comes the fool part.
# fun(ctypes.c_void_p(indata.ctypes.data), ctypes.c_void_p(outdata.ctypes.data))
fun(ctypes.c_void_p(indata.ctypes.data), ctypes.c_int(indata.shape[0]), ctypes.c_int(indata.shape[1]),ctypes.c_int(indata.shape[2]),ctypes.c_int(indata.shape[3]),
    ctypes.c_void_p(outdata.ctypes.data),ctypes.c_void_p(mask.ctypes.data),ctypes.c_void_p(index.ctypes.data))
# print ('indata: %s' % indata)
# print ('outdata: %s' % outdata)
print ('index: %s' % index)

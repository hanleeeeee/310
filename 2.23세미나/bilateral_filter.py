import cv2
import numpy as np
from PIL import Image

img = cv2.imread("D:/data/rena.jpg")



dst = cv2.bilateralFilter(img, -1, 30, 10)
dst2 = cv2.bilateralFilter(img, -1, 70, 10)


im_array = np.asarray(img) # Image to np.array

kernel = cv2.getGaussianKernel(5, 3)
res_lena_kernel1 = cv2.filter2D(img, -1, kernel)

cv2.imshow('Gaussian',res_lena_kernel1)

cv2.imshow("Lenna", img)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)

cv2.waitKey()
cv2.destroyAllWindows()

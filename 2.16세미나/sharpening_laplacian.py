import cv2
import numpy as np

image = cv2.imread('imgs/nasamoon.png',0)

#laplacian mask1
mask1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
mask2 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
mask3 = np.array([[1,1,1],[1,-8,1],[1,1,1]])
mask4 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

#scaling하지 않은 laplacian
result = cv2.filter2D(image, -1, mask1,borderType=cv2.BORDER_CONSTANT)
result2 = cv2.filter2D(image, -1, mask2,borderType=cv2.BORDER_CONSTANT)
result3 = cv2.filter2D(image, -1, mask3,borderType=cv2.BORDER_CONSTANT)
result4 = cv2.filter2D(image, -1, mask4,borderType=cv2.BORDER_CONSTANT)


sharp_img = np.clip(image.astype(np.uint64) - result.astype(np.uint64),0,255)
sharp_img = sharp_img.astype(np.uint8)

new_result = np.concatenate((image, sharp_img),axis=1)
cv2.imshow('sharpening',new_result)

new_result2 = np.concatenate((result,result2),axis=1)
cv2.imshow('result',new_result2)
cv2.waitKey(0)


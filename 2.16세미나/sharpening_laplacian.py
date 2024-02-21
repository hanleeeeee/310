import cv2
import numpy as np

image = cv2.imread('imgs/nasamoon.png',0)

#laplacian mask1
mask1 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
mask2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
mask3 = np.array([[0,1,0],[1,-4,1],[0,1,0]])

#scaling하지 않은 laplacian
laplacian1 = cv2.filter2D(image, -1, mask1,borderType=cv2.BORDER_CONSTANT)
laplacian2 = cv2.filter2D(image, -1, mask2,borderType=cv2.BORDER_CONSTANT)
laplacian3 = cv2.filter2D(image, -1, mask3,borderType=cv2.BORDER_CONSTANT)

#샤프닝 후에 normalization
sharp_img = image.astype(np.uint64) + laplacian1.astype(np.uint64)
sharp_img = np.clip(sharp_img, 0, 255)
sharp_img = sharp_img.astype(np.uint8)

#샤프닝 결과
new_image = np.concatenate((image, sharp_img),axis=1)
cv2.imshow('sharpening',new_image)

#라플라시안 결과
new_image2 = np.concatenate((laplacian1,laplacian3),axis=1)
cv2.imshow('laplacian',new_image2)
cv2.waitKey(0)


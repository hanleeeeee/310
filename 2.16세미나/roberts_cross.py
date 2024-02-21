import cv2
import numpy as np

#영상 불러오기, size 확인
image = cv2.imread('imgs/Lenna.png',0)
h,w = image.shape

#roberts cross gradient filter 만들기
kernel_x = np.array([[-1,0],[0,1]])
kernel_y = np.array([[0,1],[-1,0]])

#roberts cross gradient filter 적용
gx = cv2.filter2D(image,cv2.CV_64F,kernel_x,borderType=cv2.BORDER_REPLICATE)
gy = cv2.filter2D(image,cv2.CV_64F,kernel_y,borderType=cv2.BORDER_REPLICATE)

#밝기 값을 정수로 반올림
gx = np.abs(gx)
gy = np.abs(gy)

#kernel_x와 kernel_y 영상 조합 => kernel
gxy = np.sqrt(gx **2 + gy **2)

#정규화
gx = (gx - np.min(gx)) / (np.max(gx) - np.min(gx)) * 255
gy = (gy - np.min(gy)) / (np.max(gy) - np.min(gy)) * 255
gxy = (gxy - np.min(gxy)) / (np.max(gxy) - np.min(gxy)) * 255

#밝기 값을 정수로 반올림
gx = gx.astype(np.uint8)
gy = gy.astype(np.uint8)
gxy = gxy.astype(np.uint8)

#샤프닝 후에 normalization
sharp_img = image.astype(np.uint64) + gxy.astype(np.uint64)
sharp_img = np.clip(sharp_img, 0, 255)
sharp_img = sharp_img.astype(np.uint8)

#출력
new_img1 = np.concatenate((gx,gy,gxy),axis=1)
new_img2 = np.concatenate((image,sharp_img),axis=1)
cv2.imshow('img2',new_img1)
cv2.imshow('sharp_img',new_img2)
cv2.waitKey(0)


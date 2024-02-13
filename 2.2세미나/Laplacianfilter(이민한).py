import cv2
import numpy as np
#percentile
# def auto(img, low=1,high=99):
#     low_value=np.percentile(img,low)
#     high_value=np.percentile(img,high)
#     return low_value, high_value
img=cv2.imread("D:\data/rena.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
height,width=gray.shape
#라플라시안 필터 3가지 선별
#mask1의 경우 center가 양수이기에 기본 image에 더해주면 edge강화
#mask2의 경우 center가 음수이기에 기본 image에 빼준다.
mask1=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
mask2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
mask3=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
##라플라시안을 필터에 씌움
laplacian1=cv2.filter2D(gray,-1,mask1)
laplacian2=cv2.filter2D(gray,-1,mask2)
laplacian3=cv2.filter2D(gray,-1,mask3)
laplacian4=cv2.Laplacian(gray,-1)
#샤프팅
shape1=gray-laplacian1
shape2=gray+laplacian1


shape1 = cv2.normalize(shape1,None,0,255,cv2.NORM_MINMAX)
shape2 = cv2.normalize(shape2,None,0,255,cv2.NORM_MINMAX)

#클리핑
laplacian1=np.clip(laplacian1,0,255).astype(np.uint8)
laplacian2=np.clip(laplacian2,0,255).astype(np.uint8)
laplacian3=np.clip(laplacian3,0,255).astype(np.uint8)
laplacian4=np.clip(laplacian4,0,255).astype(np.uint8)
shape1=np.clip(shape1,0,255).astype(np.uint8)
shape2=np.clip(shape2,0,255).astype(np.uint8)

#출력
cv2.imshow("filter1",laplacian1)
cv2.imshow("filter2",laplacian2)
cv2.imshow("filter3",laplacian3)
cv2.imshow("filter4",laplacian4)
cv2.imshow("subtract",shape1)
cv2.imshow("add",shape2)
cv2.waitKey(0)
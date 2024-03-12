import cv2
import numpy as np

# 원본 이미지 경로
image_path = "D:\data/forest.png"

# 이미지를 읽어옴
img = cv2.imread(image_path)

##crayon은 아마 2.5를 줬을때 잘됐었고 forest는 0.67 일때야
#sigmoid함수
def smooth_sigmoid(x):
    # Smooth sigmoid 함수 정의
    return 0.5*(1 / (1 + np.exp(-x)))
#tanh함수
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

#sigmoid함수 이용하기 위해서 x축 범위를 -5~5로 잡았다.
norm_img = cv2.normalize(img, None, alpha=5, beta=-5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
flat = smooth_sigmoid(norm_img)

scaled_image = (flat * 255).astype(np.uint8)
scaled_image=np.clip(scaled_image,0,255)

##gamma correction함수를 썼을 경우
gamma_two_point_two = np.array(255 * (img / 255) ** 2.5, dtype='uint8')

gamma_point_four = np.array(255 * (img / 255) ** 0.67, dtype='uint8')

#adapt to sigmoid 범위를 -5~5보다 폭을 줄여서 적용
sig1=cv2.normalize(img, None, alpha=-5, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
sig1 = smooth_sigmoid(sig1)
sig1 = (sig1 * 255).astype(np.uint8)
sig2=cv2.normalize(img, None, alpha=0, beta=5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
sig2 = smooth_sigmoid(sig2)
sig2 = (sig2 * 255).astype(np.uint8)

##tanh사용 찾았다!!!!! 이걸로 gamma대신에 더 뛰어난 성능을 가진 녀석을 찾아냈어
sig3=cv2.normalize(img, None, alpha=-2, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
sig3 = tanh(sig3)###분명히 abs를 해야 값이 정확한데 안 한것이 왜 더 선명하게 나오는 걸까?
# sig3=abs(sig3)
sig3 = (sig3 * 255).astype(np.uint8)
sig4= cv2.normalize(img, None, alpha=0 ,beta=2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
sig4 = tanh(sig4)
sig4 = (sig4 * 255).astype(np.uint8)

img6=cv2.hconcat([img,scaled_image])
#pasting
img3 = cv2.hconcat([img,gamma_two_point_two, gamma_point_four])
img4=cv2.hconcat([img,sig1,sig2])
img5=cv2.hconcat([img,sig3,sig4])
#image show
cv2.imshow("flat",img6)
cv2.imshow('gamma', img3)
cv2.imshow("sigmoid",img4)
cv2.imshow('tanh',img5)
cv2.waitKey(0)

##tanh 함수 질문

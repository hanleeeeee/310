import cv2
import numpy as np

def gaussianFilter2d(width, sigma):
    # 중심에서부터의 거리 계산
    array = np.arange((width//2)*(-1), (width//2)+1)

    # 중심에서부터 거리 제곱합을 넣을 곳
    arr = np.zeros((width, width))
    for x in range(width):
        for y in range(width):
            arr[x,y] = array[x]**2+array[y]**2

    # 커널의 값을 저장할 매트릭스 생성
    kernel = np.zeros((width, width))

    for x in range(width):
        for y in range(width):
             kernel[x,y] = np.exp(-arr[x,y]/(2*sigma**2))
             # 수식에 맞게 값 저장, exp 앞 상수는 결국 normalized 될 것이기 때문에 생략

    # 전체 값의 합으로 나누어 필터 전체의 합이 1이 되도록 함
    kernel = kernel / kernel.sum()
    return kernel

#입력 영상 불러오기, size 확인
image = cv2.imread('imgs/unsharp.png',0)
h,w = image.shape

#gaussianFilter2d 함수에 width:5, sigma:3 대입 후 2차원 gaussian kernel 반환
kernel = gaussianFilter2d(31,5)
blurred = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_CONSTANT)
unsharpening = np.clip(2*image.astype(np.uint64) - blurred.astype(np.uint64),0,255).astype(np.uint8)


lst = []
for i in range(h):
    for j in range(w):
        if image[i,j] < blurred[i,j]:
            lst.append(blurred[i,j] - image[i,j])

mask = image + max(lst) - blurred

mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask)) * 255
mask = mask.astype(np.uint8)

unsharpening = np.clip(2*image.astype(np.uint64) - blurred.astype(np.uint64),0,255).astype(np.uint8)
# unsharpening = image.astype(np.uint64) + mask.astype(np.uint64)
# unsharpening = (unsharpening - np.min(unsharpening)) / (np.max(unsharpening) - np.min(unsharpening)) * 255
# unsharpening = unsharpening.astype(np.uint8)

unsharpening2 = np.clip(5.5*image.astype(np.uint64) - 4.5*blurred.astype(np.uint64),0,255).astype(np.uint8)
# unsharpening2 = image.astype(np.uint64) + mask.astype(np.uint64) * 4.5
# unsharpening2 = (unsharpening2 - np.min(unsharpening2)) / (np.max(unsharpening2) - np.min(unsharpening2)) * 255
# unsharpening2 = unsharpening2.astype(np.uint8)

# result = np.concatenate((image, blurred, mask),axis=1)
# result2 = np.concatenate((unsharpening, unsharpening2),axis=1)

result = np.concatenate((image,blurred,unsharpening),axis=1)

#결과 확인(영상)
cv2.imshow('result',result)
#cv2.imshow('result2',result2)
cv2.waitKey(0)


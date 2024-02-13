from PIL import Image
import cv2
import numpy as np

def clipping(img, min_val, max_val):
    clipped_img = img.copy()
    clipped_img[clipped_img < min_val] = min_val
    clipped_img[clipped_img > max_val] = max_val
    return clipped_img

img = cv2.imread('D:/data/city.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

print("format:", rows, cols)
arr = np.zeros((rows, cols))
arr2 = np.zeros((rows, cols))
arr3 = np.zeros((rows, cols))
arr4 = np.zeros((rows, cols))
arr5 = np.zeros((rows, cols))
arr6 = np.zeros((rows, cols))
kernel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        sum_h = np.sum(kernel_h * img[i-1:i+2, j-1:j+2])
        sum_v = np.sum(kernel_v * img[i-1:i+2, j-1:j+2])

        # Edge strength
        arr[i-1, j-1] = sum_v
        arr2[i-1, j-1] = sum_h
        # edge_strength = np.abs(sum_v) + np.abs(sum_h)
        #
        # # Store the combined edge strength
        # arr3[i - 1, j - 1] = edge_strength


arr6=arr+arr2
def get_image_bit_depth(image_path):
    image = Image.open(image_path)
    bit_depth = image.mode

    return bit_depth

# 이미지 파일 경로를 적절히 수정하세요.
image_path = 'D:\data/city.jpg'
bit_depth = get_image_bit_depth(image_path)

print(f"Image bit depth: {bit_depth}")

##sobel 함수 추가
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
sobel_mix = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=1)
##clipping
final = clipping(arr, 0, 255)
final2 = clipping(arr2, 0, 255)
final3 = clipping(sobel_vertical, 0, 255)
final4 = clipping(sobel_horizontal, 0, 255)
final5=clipping(arr6,0,255)
#함수와 이중for문의 intensity 차이 구하기
for i in range(0, rows):
    for j in range(0, cols):
        arr4[i][j]=final[i][j]-final3[i][j]
        arr5[i][j]=final2[i][j]-final4[i][j]

print(arr4)#함수와 이중포문간의 차이를 행렬로 표시 vertical
print(arr5)#함수와 이중포문간의 차이를 행렬로 표시 vertical
cv2.imshow('Original', img)
cv2.imshow('sum_v', final)
cv2.imshow('sum_h', final2)
cv2.imshow('sobel_vertical', final3)
cv2.imshow('sobel_horizontal', final4)
cv2.imshow('sobel_mix', final5)

cv2.waitKey(0)
cv2.destroyAllWindows()
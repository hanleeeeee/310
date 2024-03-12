import cv2
import numpy as np

def calculate_contrast_limits(image, percentile_low=1, percentile_high=99):
    flat_image = image.flatten()
    low_value = np.percentile(flat_image, percentile_low)
    high_value = np.percentile(flat_image, percentile_high)
    return low_value, high_value

img = cv2.imread("D:\data/rena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 라플라시안 필터 정의
mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
mask4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# 라플라시안 필터 적용
laplacian1 = cv2.filter2D(gray, -1, mask1)
laplacian2 = cv2.filter2D(gray, -1, mask2)
laplacian3 = cv2.filter2D(gray, -1, mask3)
laplacian4 = cv2.filter2D(gray, -1, mask4)

# 샤프팅
shape1 = gray - laplacian1
shape2 = gray + laplacian4
shape3 = gray - laplacian2

# 정규화
low, high = calculate_contrast_limits(gray)
shape1 = (shape1 - low) / (high - low) * 255.0
shape2 = (shape2 - low) / (high - low) * 255.0
shape3 = (shape3 - low) / (high - low) * 255.0

# 클리핑
shape1 = np.clip(shape1, low, high).astype(np.uint8)
shape2 = np.clip(shape2, low, high).astype(np.uint8)
shape3 = np.clip(shape3, low, high).astype(np.uint8)

# 출력
cv2.imshow("1", gray)
cv2.imshow("2", shape1)
cv2.imshow("3", shape2)
cv2.imshow("4", shape3)
cv2.waitKey(0)

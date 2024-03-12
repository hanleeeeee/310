import cv2
import numpy as np

img = cv2.imread("D:\data/rena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def calculate_contrast_limits(image, percentile_low=1, percentile_high=99):
    flat_image = image.flatten()
    low_value = np.percentile(flat_image, percentile_low)
    high_value = np.percentile(flat_image, percentile_high)
    return low_value, high_value
# 라플라시안 필터 정의
mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# 라플라시안 필터 적용
laplacian1 = cv2.filter2D(gray, -1, mask1)
laplacian2 = cv2.filter2D(gray, -1, mask2)
laplacian3 = cv2.filter2D(gray, -1, mask3)
laplacian4 = cv2.Laplacian(gray, -1)



low,high=calculate_contrast_limits(gray)
# 샤프팅
shape1 = gray - laplacian2
shape2 = gray + laplacian1
shape3=gray+laplacian3
# 정규화
shape1 = cv2.normalize(shape1, None, 0, 255, cv2.NORM_MINMAX)
shape2 = cv2.normalize(shape2, None, 0, 255, cv2.NORM_MINMAX)
shape3 = cv2.normalize(shape3, None, 0, 255, cv2.NORM_MINMAX)
# 클리핑
laplacian1 = np.clip(laplacian1, 0, 255).astype(np.uint8)
laplacian2 = np.clip(laplacian2, 0, 255).astype(np.uint8)
laplacian3 = np.clip(laplacian3, 0, 255).astype(np.uint8)
laplacian4 = np.clip(laplacian4, 0, 255).astype(np.uint8)
shape1 = np.clip(shape1, low,high).astype(np.uint8)
shape2 = np.clip(shape2, low,high).astype(np.uint8)
shape3=np.clip(shape3,low,high).astype(np.uint8)
cv2.imshow("subtract", shape1)
cv2.imshow("add", shape2)
cv2.imshow("forced add",shape3)
cv2.imshow("original",gray)
cv2.waitKey(0)

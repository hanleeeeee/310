# import cv2
# import numpy as np
# from PIL import Image
# def smooth_sigmoid(x):
#     # Smooth sigmoid 함수 정의
#     return (1 / (1 + np.exp(-x)))
# def rgb_to_cmyk(image_path):
#     with Image.open(image_path) as img:
#         # Convert PIL image to NumPy array
#         rgb_image = np.array(img)
#
#         # Normalize the RGB values to the range [0, 1]
#         rgb_image = rgb_image / 255.0
#
#         # Convert RGB to CMYK
#         k = 1 - np.max(rgb_image, axis=2)
#         c = (1 - rgb_image[..., 0] - k) / (1 - k)
#         m = (1 - rgb_image[..., 1] - k) / (1 - k)
#         y = (1 - rgb_image[..., 2] - k) / (1 - k)
#
#     # Convert to CMYK [0, 100]
#     cmyk_image = np.stack([c * 100, m * 100, y * 100, k * 100], axis=-1)
#     return cmyk_image.astype(np.uint8),c,m,y,k
#
# def cmyk_to_rgb(c,m,y,k):
#     # Normalize the CMYK values to the range [0, 1]
#
#
#     # Convert CMYK to RGB
#     r = 255 * (1 - c) * (1 - k)
#     g = 255 * (1 - m) * (1 - k)
#     b = 255 * (1 - y) * (1 - k)
#
#     return np.stack([r, g, b], axis=-1).astype(np.uint8)
#
# image_path = 'D:/data/momandbaby.png'
#
# cmyk_image,c,m,y,k = rgb_to_cmyk(image_path)
# m = np.array(255 * (m / 255) ** 2.5, dtype='uint8')
# rgb_image_reconstructed = cmyk_to_rgb(cmyk_image)
#
# cv2.imshow('Original Image', cv2.imread(image_path))
# cv2.imshow('Reconstructed Image', rgb_image_reconstructed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np
from PIL import Image

def smooth_sigmoid(x):
    # Smooth sigmoid 함수 정의
    return (1 / (1 + np.exp(-x)))

def rgb_to_cmyk(image_path):
    with Image.open(image_path) as img:
        # Convert PIL image to NumPy array
        rgb_image = np.array(img)

        # Normalize the RGB values to the range [0, 1]
        rgb_image = rgb_image / 255.0

        # Convert RGB to CMYK
        k = 1 - np.max(rgb_image, axis=2)
        c = (1 - rgb_image[..., 0] - k) / (1 - k)
        m = (1 - rgb_image[..., 1] - k) / (1 - k)
        y = (1 - rgb_image[..., 2] - k) / (1 - k)

    # Convert to CMYK [0, 100]
    cmyk_image = np.stack([c * 100, m * 100, y * 100, k * 100], axis=-1)
    return cmyk_image.astype(np.uint8), c, m, y, k

def cmyk_to_rgb(c, m, y, k):
    # Normalize the CMYK values to the range [0, 1]
    c, m, y, k = c / 100.0, m / 100.0, y / 100.0, k / 100.0

    # Convert CMYK to RGB
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)

    return np.stack([r, g, b], axis=-1).astype(np.uint8)

image_path = 'D:/data/momandbaby.png'

cmyk_image, c, m, y, k = rgb_to_cmyk(image_path)
m = np.array(255 * (m / 255) ** 2.5, dtype='uint8')
rgb_image_reconstructed = cmyk_to_rgb(c, m, y, k)

cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow('Reconstructed Image', rgb_image_reconstructed)
cv2.waitKey(0)
cv2.destroyAllWindows()


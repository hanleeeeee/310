import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Image Load
image = io.imread('D:\data/rubi.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sharpening function
def sharpening(image, strength):
    b = (1 - strength) / 8
    sharpening_kernel = np.array([[b, b, b],
                                  [b, strength, b],
                                  [b, b, b]])
    output = cv2.filter2D(image, -1, sharpening_kernel)
    return output

# Result
output1 = sharpening(gray_image, strength=7)
output2 = sharpening(gray_image, strength=17)

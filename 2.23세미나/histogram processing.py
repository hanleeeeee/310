import cv2
import numpy as np
from math import pi
import math
from matplotlib import pyplot as plt


# def BGR_TO_HSI(img):
#   with np.errstate(divide='ignore', invalid='ignore'):
#     # 32bit로 정규화
#     bgr = np.float32(img) / 255
#
#     # 불러오기
#     blue = bgr[:, :, 0]
#     green = bgr[:, :, 1]
#     red = bgr[:, :, 2]
#
#     # intensity
#     def calc_intensity(red, blue, green):
#       return np.divide(blue + green + red, 3)
#
#     # saturation
#     def calc_saturation(red, blue, green):
#       minimum = np.minimum(np.minimum(red, green), blue)
#       saturation = 1 - (3 / (red + green + blue) * minimum)
#
#       return saturation
#
#     # Hue
#     def calc_hue(red, blue, green):
#       hue = np.zeros_like(red, dtype=np.float32)
#
#       for i in range(0, blue.shape[0]):
#         for j in range(0, blue.shape[1]):
#           numerator = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j]))
#           denominator = float(np.sqrt((red[i][j] - green[i][j]) ** 2 + (red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
#
#           hue[i][j] = np.arccos(numerator / (denominator))
#           if blue[i][j] <= green[i][j]:
#             hue[i][j] = hue[i][j]
#           else:
#             # HUE를 라디안으로
#             hue[i][j] = 2 * np.pi - hue[i][j]
#
#       return hue
#
#     hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
#     return hsi



def BGR_TO_HSI(img):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(img)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Intensity
        def calc_intensity(red, blue, green):
          return np.divide(blue + green + red, 3)

    # Calculate Saturation
        def calc_saturation(red, blue, green):
          minimum = np.minimum(np.minimum(red, green), blue)
          saturation = 1 - (3 / (red + green + blue + 0.001) * (minimum + 0.001))  # Add small constant

          return saturation

        # Calculate Hue
        def calc_hue(red, blue, green):
          hue = np.copy(red)

          for i in range(0, blue.shape[0]):
            for j in range(0, blue.shape[1]):
              hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                          max(1e-10, math.sqrt((red[i][j] - green[i][j]) ** 2 +
                                               ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j]))))
              hue[i][j] = math.acos(min(1.0, max(-1.0, hue[i][j])))  # Ensure the value is within [-1, 1]

              if blue[i][j] <= green[i][j]:
                hue[i][j] = hue[i][j]
              else:
                hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

          return hue

        #Merge channels into picture and return image
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
        return hsi
def HSI_TO_BGR(img):
    with np.errstate(divide='ignore', invalid='ignore'):
      hsi = np.float32(cv2.split(img))
      hue = hsi[0]
      saturation = hsi[1]
      intensity = hsi[2]

      # 초기화
      blue = np.zeros_like(hue, dtype=np.float64)
      green = np.zeros_like(hue, dtype=np.float64)
      red = np.zeros_like(hue, dtype=np.float64)
      print(hue.shape[0])
      print(hue.shape[1])
      # 2중 포문을 통해 HUE값을 픽셀별로 접근
      for i in range(0, hue.shape[0]):
        for j in range(0, hue.shape[1]):
          # HUE를 라디안으로 표현하기 위해서 정규화
          hue_normalized = (hue[i][j] * 180) / pi
          # 경계별로 조건문 시행
          if (0 <= hue_normalized) and (hue_normalized < 120):
            hue_normalized = hue_normalized / 180 * pi
            blue[i][j] = intensity[i][j] * (1 - saturation[i][j])
            red[i][j] = intensity[i][j] * (
                      1 + (saturation[i][j] * np.cos(hue_normalized) / np.cos(pi / 3 - hue_normalized)))
            green[i][j] = 3 * intensity[i][j] - (red[i][j] + blue[i][j])
          elif (120 <= hue_normalized) and (hue_normalized < 240):
            hue_normalized = (hue_normalized - 120) / 180 * pi
            red[i][j] = intensity[i][j] * (1 - saturation[i][j])
            green[i][j] = intensity[i][j] * (
                      1 + (saturation[i][j] * np.cos(hue_normalized)) / (np.cos(pi / 3 - hue_normalized)))
            blue[i][j] = 3 * intensity[i][j] - (red[i][j] + green[i][j])
          elif (240 <= hue_normalized) and (hue_normalized <= 360):
            hue_normalized = (hue_normalized - 240) / 180 * pi
            green[i][j] = intensity[i][j] * (1 - saturation[i][j])
            blue[i][j] = intensity[i][j] * (
                      1 + (saturation[i][j] * np.cos(hue_normalized)) / (np.cos(pi / 3 - hue_normalized)))
            red[i][j] = 3 * intensity[i][j] - (green[i][j] + blue[i][j])

      bgr = cv2.merge((blue, green, red))
      return bgr
img=cv2.imread("D:\data/rena.jpg")
hsi=BGR_TO_HSI(img)
hue,saturation,intensity=cv2.split(hsi)
gamma_two_point_two = np.array(255 * (intensity / 255) ** 0.67, dtype='float32')
gamma_two_point_two=cv2.normalize(gamma_two_point_two,None,0,1,cv2.NORM_MINMAX)
print(hue.dtype)
print(gamma_two_point_two.dtype)
#intensity2=cv2.equalizeHist(intensity)
hsi2=cv2.merge((hue,saturation,gamma_two_point_two))
rgb=HSI_TO_BGR(hsi2)

cv2.imshow("original",img)
cv2.imshow("reconstrcted",rgb)

cv2.waitKey(0)
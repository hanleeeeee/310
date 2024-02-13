import cv2
import numpy as np
import matplotlib.pyplot as plt
##contrast_stretching과정
def contrast_stretching(img,newmin,newmax):
    ch_img=img.copy()
    ch_img[ch_img<=newmin]=newmin
    ch_img[ch_img>=newmax]=newmax
    result=(ch_img-newmin)/(newmax-newmin)*255
    return result
#histogram 설계과정
def show(img,max_ylim=12500):
    plt.imshow(img,cmap='gray')
    plt.show()
    if max_ylim!='none':
        axes=plt.axes()
        axes.set_ylim([0,max_ylim])
    plt.hist(img.ravel(),bins=256,range=[0,256])
    plt.show()
#high, low값 받아오기
def auto_contrast_stretching(img, percentile_low=1,percentile_high=99):
    low_value,high_value=calculate_contrast_limits(img,percentile_low,percentile_high)
    result=contrast_stretching(img,newmin=low_value,newmax=high_value)

    return result
#상위 99%, 하위1% intensity구하는 과정
def calculate_contrast_limits(image,percentile_low=1,percentile_high=99):
    flat_image=image.flatten()
    low_value=np.percentile(flat_image,percentile_low)
    high_value=np.percentile(flat_image,percentile_high)
    return low_value,high_value
img=cv2.imread("D:\data/rena.jpg")
img_1=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
npimg_1=np.array(img_1)
#기존의 rena
show(npimg_1,max_ylim=12500)
##contrast_stretching된 결과
stimg_1=auto_contrast_stretching(npimg_1)
show(stimg_1,max_ylim=12500)
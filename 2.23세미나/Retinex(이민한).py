# import numpy as np
# import cv2
#
#
# def singleScaleRetinex(img, variance):
#     retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
#     return retinex
#
#
# def multiScaleRetinex(img, variance_list):
#     retinex = np.zeros_like(img)
#     for variance in variance_list:
#         retinex += singleScaleRetinex(img, variance)
#     retinex = retinex / len(variance_list)
#     return retinex
#
#
# def MSR(img, variance_list):
#     img = np.float64(img) + 1.0
#     img_retinex = multiScaleRetinex(img, variance_list)
#
#     for i in range(img_retinex.shape[2]):
#         unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
#         for u, c in zip(unique, count):
#             if u == 0:
#                 zero_count = c
#                 break
#         low_val = unique[0] / 100.0
#         high_val = unique[-1] / 100.0
#         for u, c in zip(unique, count):
#             if u < 0 and c < zero_count * 0.1:
#                 low_val = u / 100.0
#             if u > 0 and c < zero_count * 0.1:
#                 high_val = u / 100.0
#                 break
#         img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
#
#         img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
#                                (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
#                                * 255
#     img_retinex = np.uint8(img_retinex)
#     return img_retinex
#
#
# def SSR(img, variance):
#     img = np.float64(img) + 1.0
#     img_retinex = singleScaleRetinex(img, variance)
#     for i in range(img_retinex.shape[2]):
#         unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
#         for u, c in zip(unique, count):
#             if u == 0:
#                 zero_count = c
#                 break
#         low_val = unique[0] / 100.0
#         high_val = unique[-1] / 100.0
#         for u, c in zip(unique, count):
#             if u < 0 and c < zero_count * 0.1:
#                 low_val = u / 100.0
#             if u > 0 and c < zero_count * 0.1:
#                 high_val = u / 100.0
#                 break
#         img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
#
#         img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
#                                (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
#                                * 255
#     img_retinex = np.uint8(img_retinex)
#     return img_retinex
#
#
# variance_list = [15, 80, 250]
# variance = 300
#
# img = cv2.imread('D:\data/kyeong.png')
# img_msr = MSR(img, variance_list)
# img_ssr = SSR(img, variance)
#
# cv2.imshow('Original', img)
# cv2.imshow('MSR', img_msr)
# cv2.imshow('SSR', img_ssr)
# cv2.imwrite('SSR.jpg', img_ssr)
# cv2.imwrite('MSR.jpg', img_msr)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import numpy as np
import cv2
from PIL import Image


def get_ksize(sigma):
    # opencv calculates ksize from sigma as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # then ksize from sigma is
    # ksize = ((sigma - 0.8)/0.15) + 2.0

    return int(((sigma - 0.8) / 0.15) + 2.0)


def get_gaussian_blur(img, ksize=0, sigma=5):
    # if ksize == 0, then compute ksize from sigma
    if ksize == 0:
        ksize = get_ksize(sigma)

    # Gaussian 2D-kernel can be seperable into 2-orthogonal vectors
    # then compute full kernel by taking outer product or simply mul(V, V.T)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    # if ksize >= 11, then convolution is computed by applying fourier transform
    imge=cv2.filter2D(img+1.0, -1, np.outer(sep_k, sep_k))
    imge=np.clip(imge,0,255)
    return (imge)



def ssr(img, sigma):
    img = img + 1.0  # 0이 들어가지 않도록 1을 더함

    return np.log10(img) - np.log10(get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)


def msr(img, sigma_scales=[15, 80, 250]):
    # Multi-scale retinex of an image
    # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales

    msr = np.zeros(img.shape)
    # for each sigma scale compute SSR
    for sigma in sigma_scales:
        msr += ssr(img, sigma)

    # divide MSR by weights of each scale
    # here we use equal weights
    msr = msr / len(sigma_scales)

    # computed MSR could be in range [-k, +l], k and l could be any real value
    # so normalize the MSR image values in range [0, 255]
    msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    return msr



def color_balance(img, low_per, high_per):
    '''Contrast stretch img by histogram equilization with black and white cap'''

    tot_pix = img.shape[1] * img.shape[0]
    # no.of pixels to black-out and white-out
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100

    # channels of image
    ch_list = []
    if len(img.shape) == 2:
        ch_list = [img]
    else:
        ch_list = cv2.split(img)

    cs_img = []
    # for each channel, apply contrast-stretch
    for i in range(len(ch_list)):
        ch = ch_list[i]
        # cummulative histogram sum of channel
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))

        # find indices for blacking and whiting out pixels
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if (li == hi):
            cs_img.append(ch)
            continue
        # lut with min-max normalization for [0-255] bins
        lut = np.array([0 if i < li
                        else (255 if i > hi else round((i - li) / (hi - li) * 255))
                        for i in np.arange(0, 256)], dtype='uint8')
        # constrast-stretch channel
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)

    if len(cs_img) == 1:
        return np.squeeze(cs_img)
    elif len(cs_img) > 1:
        return cv2.merge(cs_img)
    return None


def msrcr(img, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
    # Multi-scale retinex with Color Restoration
    # MSRCR(x,y) = G * [MSR(x,y)*CRF(x,y) - b], G=gain and b=offset
    # CRF(x,y) = beta*[log(alpha*I(x,y) - log(I'(x,y))]
    # I'(x,y) = sum(Ic(x,y)), c={0...k-1}, k=no.of channels

    img = img.astype(np.float64) + 1.0
    # Multi-scale retinex and don't normalize the output
    msr_img = msr(img, sigma_scales, apply_normalization=False)
    # Color-restoration function
    crf = beta * (np.log10(alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))
    # MSRCR
    msrcr = G * (msr_img * crf - b)
    # normalize MSRCR
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    # color balance the final MSRCR to flat the histogram distribution with tails on both sides
    msrcr = color_balance(msrcr, low_per, high_per)

    return msrcr


def msrcp(img, sigma_scales=[15, 80, 250], low_per=1, high_per=1):

    # Intensity image (Int)
    int_img = (np.sum(img, axis=2) / img.shape[2]) + 1.0
    # Multi-scale retinex of intensity image (MSR)
    msr_int = msr(int_img, sigma_scales)
    # color balance of MSR
    msr_cb = color_balance(msr_int, low_per, high_per)

    # B = MAX/max(Ic)
    B = 256.0 / (np.max(img, axis=2) + 1.0)
    # BB = stack(B, MSR/Int)
    BB = np.array([B, msr_cb / int_img])
    # A = min(BB)
    A = np.min(BB, axis=0)
    # MSRCP = A*I
    msrcp = np.clip(np.expand_dims(A, 2) * img, 0.0, 255.0)
    return msrcp.astype(np.uint8)

im = Image.open('D:/data/greek.jpg')
width, height = im.size
print(width, height)
img = cv2.imread('D:/data/greek.jpg')

img_ssr = cv2.resize(ssr(img, 15),(width, height))
img_ssr=np.clip(img_ssr,0,255)
img_ssr1 = cv2.resize(ssr(img, 80),(width, height))
img_ssr2 = cv2.resize(ssr(img, 250),(width, height))
img_msr=cv2.resize(msr(img),(width, height))
img_msrcr=cv2.resize(msr(img),(width, height))
img_msrcp=cv2.resize(msrcp(img),(width, height))

cv2.imshow("original", img)
# cv2.imshow('SSR15', img_ssr)
cv2.imshow('SSR80', img_ssr1)
# cv2.imshow('SSR250', img_ssr2)
cv2.imshow("MSR",img_msr)
cv2.imshow("MSRCR",img_msrcr)
cv2.imshow("MSRCP",img_msrcp)

cv2.waitKey(0)
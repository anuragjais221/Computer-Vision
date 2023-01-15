# -*- coding: utf-8 -*-
"""2021csm1002_cv_assignment01.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hCQ1v1bUU2FooctOvkbVpx3WFM1Tp2de
"""

# !python --version

"""## Task1:
#Write your own Canny edge detector – 10 marks
"""




from math import log10, sqrt
from skimage import io
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np

from skimage import feature # For canny()
from skimage.metrics import structural_similarity as ssim
import os
from os import listdir


def gaussian_kernel(size, sigma=1.5):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def my_convolve_pad(img, gaussian_filter_2d):
    gaussian_2d_out = []
    filter_size = gaussian_filter_2d.shape[0]
    k = int((filter_size - 1)/2)
    pad_img = np.pad(img, (k, ), 'constant',
                     constant_values=(0, 0))

    # print(pad_img.shape)
    for i in range(k, pad_img.shape[0]-k):
        temp = []
        for j in range(k, pad_img.shape[1]-k):
            mat = pad_img[i-k:i+k+1, j-k:j+k+1]
            temp.append(np.sum(gaussian_filter_2d * mat))
        gaussian_2d_out.append(temp)
    return np.array(gaussian_2d_out)


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = my_convolve_pad(img, Kx)
    Iy = my_convolve_pad(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def non_max_suppression(img, D):
    m, n = img.shape
    nonMaxSuppImg = np.zeros((m, n), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, m-1):
        for j in range(1, n-1):

            qpixel = 255
            rpixel = 255

            # angle 135
            if (112.5 <= angle[i, j] < 157.5):
                qpixel = img[i-1, j-1]
                rpixel = img[i+1, j+1]
            # angle 0
            elif (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                qpixel = img[i, j+1]
                rpixel = img[i, j-1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                qpixel = img[i+1, j-1]
                rpixel = img[i-1, j+1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                qpixel = img[i+1, j]
                rpixel = img[i-1, j]

            if (img[i, j] >= qpixel) and (img[i, j] >= rpixel):
                nonMaxSuppImg[i, j] = img[i, j]
            else:
                nonMaxSuppImg[i, j] = 0

    return nonMaxSuppImg


def threshold(img, lowThresholdRatio=0.1, highThresholdRatio=0.2, weak_pixel=30, strong_pixel=255,):

    # highThreshold = img.max() * highThresholdRatio;
    # lowThreshold = highThreshold * lowThresholdRatio;
    diff = img.max()-img.min()
    highThreshold = img.min() + diff * highThresholdRatio
    lowThreshold = img.min() + diff * lowThresholdRatio
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)
    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res


def assignment(i, j, img, strong):
    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):

        return strong

    return 0


def hysteresis(img, weak_pixel=30, strong_pixel=255):
    M, N = img.shape
    weak = weak_pixel  # weak pixel
    strong = strong_pixel  # strong pixel
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                img[i, j] = assignment(i, j, img, strong)
                # try:
                #     if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                #         or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                #         or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                #         img[i, j] = strong
                #     else:
                #         img[i, j] = 0
                # except IndexError as e:
                #     pass
    return img


kernel_size = 5


def myCannyEdgeDetector(image, grayImage,imgName):
    ''' STEP 1 : Apply gaussian filter to reduce the noise from the image '''
    smooth_image = my_convolve_pad(grayImage, gaussian_kernel(kernel_size))
    ''' STEP 2 : Find the edge intenity and direction by calculating the gradient of the image using edge detection operators '''
    gradientMat, thetaMat = sobel_filters(smooth_image)
    # plt.imshow(gradientMat, cmap='gray')
    # plt.imshow(thetaMat, cmap='gray')
    ''' STEP 3 : Non-Maximum Suppression '''
    nonMaxImg = non_max_suppression(gradientMat, thetaMat)
    ''' STEP 4 : Linking and Thresholding '''
    thresholdImg = threshold(nonMaxImg)
    final_img = hysteresis(thresholdImg)
    fig, axarr = plt.subplots(3, 3, figsize=(20, 15))
    axarr[0, 0].imshow(image, cmap='gray')
    axarr[0, 0].set_title('Input Image')
    axarr[0, 0].axis('off')
    axarr[0, 1].imshow(grayImage, cmap='gray')
    axarr[0, 1].set_title('Grayscale Image')
    axarr[0, 1].axis('off')
    axarr[0, 2].imshow(smooth_image, cmap='gray')
    axarr[0, 2].set_title('Smoothen Image')
    axarr[0, 2].axis('off')
    axarr[1, 0].imshow(gradientMat, cmap='gray')
    axarr[1, 0].set_title('GradientMat')
    axarr[1, 0].axis('off')
    axarr[1, 1].imshow(thetaMat, cmap='gray')
    axarr[1, 1].set_title('Theta Mat')
    axarr[1, 1].axis('off')
    axarr[1, 2].imshow(nonMaxImg, cmap='gray')
    axarr[1, 2].set_title('Non Maximal Supression')
    axarr[1, 2].axis('off')
    axarr[2, 0].imshow(thresholdImg, cmap='gray')
    axarr[2, 0].set_title('Linking & Thresholding')
    axarr[2, 0].axis('off')
    axarr[2, 1].imshow(final_img, cmap='gray')
    axarr[2, 1].set_title('Hysteresis')
    axarr[2, 1].axis('off')
    axarr[2, 2].imshow(final_img, cmap='gray')
    axarr[2, 2].set_title('Final Image')
    axarr[2, 2].axis('off')
    fig.savefig("canny"+imgName)
    return final_img


"""### Compute the peak signal to noise ratio (PSNR)"""
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# Toy_image PSNR 17.82
# test.jpg 12.466061202114187
# Einstein 11.147959617274424
# bird 16.369293269452555
# bicycle 10.33165670909931
# dog 14.032288326101426
# 4.jpg 16.7235187893353
# astronout  12.615122328435124
# 1.jpg 17.521831217119793


# Toy_image 0.9201
# Test jpg
# Einstein 0.6075388754630748
# bird 0.8649599248972781
# bicycle 0.5326828548340671
# dog 0.7295087517970784
# 4.jpg 0.8419373937745458
# astronaut 0.6531842178510496
# 1.jpg 0.8855775083101166
input_path = "./images/"
# output_path = "./canny_output/"


def skImageCannyDetector(grayImage,imgName):
    smooth_image = my_convolve_pad(grayImage, gaussian_kernel(kernel_size))
    edgeMapCanny = feature.canny(smooth_image)
    fig = plt.figure()
    plt.imshow(edgeMapCanny, cmap='gray')
    fig.suptitle('Skimage canny output')
    plt.savefig('SkImage'+imgName)
    return edgeMapCanny


def load_data(input_path):
    imgs=[]
    for images in os.listdir(input_path):
    
        # check if the image ends with png or jpg or jpeg
        if (images.endswith(".png") or images.endswith(".jpg")\
            or images.endswith(".jpeg") or images.endswith(".bmp")):
            imgs.append(images)

    return imgs

def MyCannyEdgeDetectorDemo():
    
    images = load_data(input_path)
    print(images)
    for img in images:
        image = io.imread(input_path+img)        
        grayImage = rgb2gray(image)
        final_img = myCannyEdgeDetector(image, grayImage, img)
        edgeMapCanny = skImageCannyDetector(grayImage,img)
        plt.show()
        # PSNR Calculation
        value = PSNR(edgeMapCanny, final_img)
        print("PSNR value : ", value)
        # Calculating SSIM
        ssim_val = ssim(edgeMapCanny, final_img, gaussian_weights=True,
                        sigma=1.5, use_sample_covariance=False, data_range=1.0)
        print("SSIM val : ", ssim_val)


MyCannyEdgeDetectorDemo()



### References :

''' https://en.wikipedia.org/wiki/Sobel_operator

https://medium.com/@enzoftware/how-to-build-amazing-images-filters-with-python-median-filter-sobel-filter-%EF%B8%8F-%EF%B8%8F-22aeb8e2f540

https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm

https://www.geeksforgeeks.org/python-grayscaling-of-images-using-opencv

https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html

https://learnopencv.com/edge-detection-using-opencv/ '''
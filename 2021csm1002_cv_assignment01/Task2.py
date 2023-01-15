from skimage import io
from skimage.color import rgb2gray
from skimage import io
from skimage.color import rgb2gray
from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
import cv2

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

def BlurOrNot(grayImage,threshold=120):
    
    grayImage = grayImage*255
    Lx = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],np.float32)
    # Lx = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]],np.float32)
    # blur = laplace(grayImage,ksize=3)
    blur = my_convolve_pad(grayImage,Lx)
    score = np.var(blur)
    return blur,score,score < threshold


input_path = "./images/"

def load_data(input_path):
    imgs=[]
    for images in os.listdir(input_path):
    
        # check if the image ends with png or jpg or jpeg
        if (images.endswith(".png") or images.endswith(".jpg")\
            or images.endswith(".jpeg") or images.endswith(".bmp")):
            imgs.append(images)

    return imgs


def task2():
    images = load_data(input_path)
    print(images)
    scores=[]
    isblurs=[]
    for img in images:
        image = io.imread(input_path+img)
        grayImage = rgb2gray(image)
        # grayImage = grayImage*255
        # grayImage = resize(grayImage, (256, 256), preserve_range=True, anti_aliasing=False)
        # pixel_brightness = []
        # for x in range (1,480):
        #     for y in range (1,640):
        #         try:
        #             pixel = image[x,y]
        #             R, G, B = pixel
        #             brightness = sum([R,G,B])/3
        #             pixel_brightness.append(brightness)
        #         except IndexError:
        #             pass
        # din_range = round(np.log2(max(pixel_brightness))-np.log2(min((pixel_brightness))), 2)
        # # dinamic_range.append(din_range)
        # print('The image', img, 'has a dinamic range of', din_range, 'EV')
        # threshold = din_range*0.25  
        # print('threshold',threshold)      
        blur,score,isBlur = BlurOrNot(grayImage) 
        scores.append(score/255)
        isblurs.append(isBlur)
    
    mini = min(scores)
    maxi = max(scores)
    print('scores',scores)
    print("max",maxi)
    print("min",mini)

    probalilities=[]
    size = len(scores)
    # for s in scores: 
    for i in range(size):
        # if isblurs[i] == False:
        prob = ( scores[i] - mini )/ (maxi-mini)
        # else:
            # prob = ( scores[i] - mini )/ (maxi-mini)
        probalilities.append(prob)

    print('probablities',probalilities)

    
    # fig, axarr = plt.subplots(3, 3, figsize=(20, 15))

    for i in range(size):
        image = io.imread(input_path+images[i])
        plt.imshow(image)
        plt.figtext(0, 0, "Image Blur :" + str(isblurs[i]) + "score : " + str(score) + " Normalized_score : " + str(probalilities[i]) , fontsize = 10,bbox={"facecolor":"orange", "alpha":0.5})
        plt.show()
    

task2()
# image = io.imread("./images/London_Blur.jpg")
# pixel_brightness = []
# for x in range (1,480):
#     for y in range (1,640):
#         try:
#             pixel = image[x,y]
#             R, G, B = pixel
#             brightness = sum([R,G,B])/3
#             pixel_brightness.append(brightness)
#         except IndexError:
#             pass
# din_range = round(np.log2(max(pixel_brightness))-np.log2(min((pixel_brightness))), 2)
# # dinamic_range.append(din_range)
# # print('The image', img, 'has a dinamic range of', din_range, 'EV')
# print('din_range',din_range)
# threshold = din_range*0.25  
# print('threshold',threshold)      
# d = rgb2gray(image)
# m,n = d.shape 
# for x in range(1,m-1):
#     for y in range(1,n-1):
#         d[x,y] = max(abs(2*d[x,y] - d[x,y+1] -d[x,y-1]), abs(2*d[x,y] - d[x+1,y] -d[x-1,y]))




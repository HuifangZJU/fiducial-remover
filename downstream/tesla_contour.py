import os,csv,re, time
import pickle
import random
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
from scipy.sparse import issparse
import scanpy as sc
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
import TESLA as tesla
from IPython.display import Image
# print(tesla.__version__)
from PIL import Image

def get_rgb_img(path):
    # Open the RGBA image
    rgba_image = Image.open(path).convert("RGBA")

    # Convert RGBA image to a NumPy array
    rgba_array = np.array(rgba_image)

    # Extract the alpha channel
    alpha_channel = rgba_array[..., 3]

    # Set the alpha channel to binary: fully transparent (0) or fully opaque (255)
    binary_alpha = np.where(alpha_channel > 200, 255, 0).astype(np.uint8)

    # Replace the original alpha channel with the binary alpha
    rgba_array[..., 3] = binary_alpha



    # Convert the modified NumPy array back to a PIL RGBA image
    binary_rgba_image = Image.fromarray(rgba_array, 'RGBA')

    # Create a white background image (RGBA)
    white_background = Image.new('RGBA', binary_rgba_image.size, (255, 255, 255, 255))

    # Blend the binary RGBA image with the white background
    blended_image = Image.alpha_composite(white_background, binary_rgba_image).convert('RGB')

    # Convert the blended image to a NumPy array and return
    rgb_array = np.array(blended_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


# imglist = '/home/huifang/workspace/data/imagelists/tiff_img_list.txt'
# file = open(imglist)
# lines = file.readlines()
# for i in range(20,len(lines)):
#     print(i)
#     line = lines[i]
#     filename = line.rstrip().split(' ')[0]
#     img=cv2.imread(filename)

imgfolder = '/media/huifang/data/fiducial/tiff/recovered_tiff/'
for i in range(3,20):
    print(i)
    filename = imgfolder+str(i)+'_cleaned.png'
    filename = "/media/huifang/data/fiducial/tiff_data/V1_Mouse_Brain_Sagittal_Anterior_Section_2_spatial/V1_Mouse_Brain_Sagittal_Anterior_Section_2_image_cleaned.png"
    img = get_rgb_img(filename)

    resize_factor=1000/np.min(img.shape[0:2])
    resize_width=int(img.shape[1]*resize_factor)
    resize_height=int(img.shape[0]*resize_factor)
    cnt=tesla.cv2_detect_contour(img, apertureSize=5,L2gradient = True)
    binary=np.zeros((img.shape[0:2]), dtype=np.uint8)
    cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
    #Enlarged filter
    cnt_enlarged = tesla.scale_contour(cnt, 1.05)
    binary_enlarged = np.zeros(img.shape[0:2])
    cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
    img_new = img.copy()
    cv2.drawContours(img_new, [cnt], -1, (255), thickness=50)
    img_new=cv2.resize(img_new, ((resize_width, resize_height)))
    img_new_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR)
    # plt.imshow(img_new_new)
    # plt.show()

    cv2.imwrite(filename[:-4]+'_tesla_contour.jpg', img_new)


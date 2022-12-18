from __future__ import division
import os.path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.measure import label
from hough_utils import *
from fiducial_utils import read_tissue_image as read_image

crop_size = 16
augsize =6
circle_width = 2

def get_circle_pixels(circle_center,radius):
    x0 = circle_center[0]
    y0 = circle_center[1]
    pixels=[]

    x = radius
    y = 0
    while (y < x):
        pixels.append([x + x0, y + y0])
        pixels.append([y + x0, x + y0])
        pixels.append([-x + x0, y + y0])
        pixels.append([-y + x0, x + y0])
        pixels.append([-x + x0, -y + y0])
        pixels.append([-y + x0, -x + y0])
        pixels.append([x + x0, -y + y0])
        pixels.append([y + x0, -x + y0])
        if np.power(x, 2) + np.power(y + 1, 2) > np.power(radius, 2):
            x = x - 1
        y = y + 1
    pixels = np.asarray(pixels)
    return [pixels[:,0],pixels[:,1]]

def write_image_in_dir_to_file(root_dir,f,ds_rate,save_in_tissue=True,save_out_tissue=True):
    image_names = os.listdir(root_dir)
    for image_name in image_names:
        dir = root_dir + image_name+'/'
        if save_in_tissue:
            in_tissue_images = os.listdir(dir+'in_tissue')
            for crop_image_name in in_tissue_images:
                if crop_image_name.endswith('image.png'):
                    crop_image = dir+'in_tissue/'+crop_image_name
                    crop_image_mask = crop_image[:-9]+'mask'+crop_image[-4:]
                    f.write(crop_image+' '+crop_image_mask + '\n')
        if save_out_tissue:
            out_tissue_images = os.listdir(dir + 'out_tissue')
            i = 0
            for crop_image_name in out_tissue_images:
                if crop_image_name.endswith('image.png'):
                    i += 1
                    if np.mod(i,ds_rate)!=0:
                        continue
                    crop_image = dir + 'out_tissue/' + crop_image_name
                    crop_image_mask = crop_image[:-9] + 'mask' + crop_image[-4:]
                    f.write(crop_image + ' ' + crop_image_mask + '\n')
    # print(dir +' done')


def write_to_file(image_root_path,save_file,downsample_rate,save_in_tissue=True,save_out_tissue=True):

    f = open(save_file,'w')
    image_second_paths = os.listdir(image_root_path)
    for image_path in image_second_paths:
        temp_path = image_root_path + image_path + '/'
        write_image_in_dir_to_file(temp_path,f,downsample_rate,save_in_tissue,save_out_tissue)






def save_local_crop(image,mask,circle,path,iter_num=1,augmentation=True):

    for i in range(iter_num):
        xc = circle[0]
        yc = circle[1]
        if augmentation:
            x_rand = random.randint(-augsize, augsize)
            y_rand = random.randint(-augsize, augsize)
            xc = xc+x_rand
            yc = yc+y_rand
        if yc >crop_size and xc>crop_size and yc+crop_size < mask.shape[0] and xc+crop_size<mask.shape[1]:
            crop_image = image[yc - crop_size:yc + crop_size, xc - crop_size:xc + crop_size, :]
            crop_mask = mask[yc - crop_size:yc + crop_size, xc - crop_size:xc + crop_size]
            # f,a = plt.subplots(1,2)
            # a[0].imshow(crop_image)
            # a[1].imshow(crop_mask)
            # plt.show()
            save_image(crop_image, path + str(yc) + '_' + str(xc) + '_'+ str(i)+'_image.png',format="RGB")
            save_image(crop_mask, path + str(yc) + '_' + str(xc) + '_'+ str(i)+'_mask.png', format="L")

def remove_isolated_area(input_image, size):
    input_image_comp = cv2.bitwise_not(input_image)  # could just use 255-img
    kernel1 = np.zeros([3,3])
    kernel1[1:-1, 1:-1] = 1
    kernel1 = kernel1.astype(np.uint8)


    kernel2 = np.ones([size, size])
    kernel2[1:-1, 1:-1] = 0
    kernel2 = kernel2.astype(np.uint8)

    # test = cv2.filter2D(input_image,-1,kernel1)
    # plt.imshow(test)
    # plt.show()

    hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_DILATE, kernel1)
    hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
    # hitormiss1 = cv2.filter2D(input_image,-1,kernel1)
    # hitormiss2 = cv2.filter2D(input_image,-1,kernel2)
    hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
    hitormiss_comp = cv2.bitwise_not(hitormiss)  # could just use 255-img
    del_isolated = cv2.bitwise_and(input_image, input_image, mask=hitormiss_comp)
    # f, a = plt.subplots(2, 3)
    # a[0, 0].imshow(hitormiss1)
    # a[0, 1].imshow(hitormiss2)
    # a[0, 2].imshow(hitormiss)
    # a[1, 0].imshow(input_image)
    # a[1, 1].imshow(del_isolated)
    #
    # plt.show()
    return del_isolated

def get_edge_pixels(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(thresh, 1000, 1000)

    kernel_d = np.ones((3, 3), dtype=np.uint8)
    kernel_d[0,0]=0
    kernel_d[0,2]=0
    kernel_d[2,0]=0
    kernel_d[2,2]=0
    kernel_e = np.ones((3, 3), dtype=np.uint8)

    edges_temp = cv2.erode(edges,kernel_d,iterations=1)
    edges = edges - edges_temp


    edges = cv2.dilate(edges, kernel_d, iterations=2)

    #
    edges = label(edges, neighbors=8, connectivity=3)
    re = morphology.remove_small_objects(edges, min_size=30000)
    re = np.where(re > 0, 255, 0)
    re = re.astype(np.uint8)


    # f,a = plt.subplots(1,2)
    # a[0].imshow(edges)
    # a[1].imshow(re)
    # plt.show()
    edges = cv2.erode(re, kernel_d, iterations=2)
    edges_temp = cv2.erode(edges, kernel_d, iterations=1)
    re = edges - edges_temp



    # plt.imshow(re)
    # plt.show()
    #
    # edges = label(edges, neighbors=4)
    # edges = morphology.remove_small_objects(edges, min_size=6)
    # edges = np.where(edges > 0, 255, 0)
    # edges = edges.astype(np.uint8)
    #
    # # edges = cv2.erode(edges, kernel_e, iterations=1)
    # # edges = cv2.dilate(edges, kernel_d, iterations=1)
    #
    # plt.imshow(edges)
    # plt.show()
    #
    #
    # plt.imshow(edges)
    # plt.show()
    #
    #
    #
    # edges = cv2.dilate(edges, kernel_d, iterations=2)
    # edges = cv2.erode(edges,kernel_e,iterations=1)
    # edges = label(edges, neighbors=4)
    # edges = morphology.remove_small_objects(edges, min_size=8)
    # edges = np.where(edges > 0, 255, 0)
    # edges = edges.astype(np.uint8)
    # plt.imshow(edges)
    # plt.show()
    #
    #
    #
    # plt.imshow(edges)
    # plt.show()


    # edges = cv2.Canny(thresh,1200,1200)


    return re


# ------------------------------------------
#                data loading
# ------------------------------------------

root_path = '/home/huifang/workspace/data/fiducial_train/'
save_root = '/home/huifang/workspace/data/fiducial_crop_w2_aug'+str(augsize)+'_iter4/'
dataset_names = ['mouse','humanpilot']

for dataset_name in dataset_names:
    print('-------------'+dataset_name+'-------------')
    image_names= os.listdir(root_path+dataset_name)
    for image_name in image_names:
        print(image_name+'...')
        tissue_image = read_image(root_path+dataset_name+'/' + image_name)
        image = cv2.imread(root_path+dataset_name+'/' + image_name+'/masks/human_in_loop_mask_solid_result.png')
        negatives = get_edge_pixels(image)
        # f,a = plt.subplots(1,2)
        # a[0].imshow(tissue_image)
        # a[1].imshow(negatives)
        # plt.show()
        negative_position = np.where(negatives>0)
        num_position = len(negative_position[0])
        print(len(negative_position[0]))
        mask = np.zeros(image.shape)
        negative_save_path = save_root+dataset_name+'/' + image_name +'/negative/'
        os.makedirs(negative_save_path,exist_ok=True)

        for id in range(0,num_position,10):
            save_local_crop(tissue_image, mask, [negative_position[1][id],negative_position[0][id]],negative_save_path,iter_num=1,augmentation=False)
        print('done')

#
# save_file = '/home/huifang/workspace/data/imagelists/fiducial_width'+ str(circle_width)+'_all.txt'
# write_to_file(save_root,save_file,1)

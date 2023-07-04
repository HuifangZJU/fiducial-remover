from __future__ import division
import os.path
import random
import matplotlib.pyplot as plt
import os

import numpy as np

from hough_utils import *
from fiducial_utils import read_tissue_image as read_image

crop_size = 16
augsize =6
circle_width = 2

def write_image_list_to_file(f,in_tissue_list, out_tissue_list,negative_list, save_in_tissue=True,save_out_tissue=True,save_negative=True,downsample=True,downsample_base=0):
    if downsample:
        in_tissue_ds_rate = np.floor(len(in_tissue_list)/downsample_base)
        out_tissue_ds_rate = np.floor(len(out_tissue_list)/downsample_base)
        negative_ds_rate = np.floor(len(negative_list)/downsample_base)*8
    else:
        in_tissue_ds_rate = 1
        out_tissue_ds_rate = 1
        negative_ds_rate = 1

    print(in_tissue_ds_rate)
    print(out_tissue_ds_rate)
    print(negative_ds_rate)
    test = input()
    if save_in_tissue:
        i = 0
        for crop_image in in_tissue_list:
            i += 1
            if np.mod(i, in_tissue_ds_rate) != 0:
                continue
            crop_image_mask = crop_image[:-9] + 'mask' + crop_image[-4:]
            f.write(crop_image + ' ' + crop_image_mask + '\n')
    if save_out_tissue:
        i = 0
        for crop_image in out_tissue_list:
            i += 1
            if np.mod(i, out_tissue_ds_rate) != 0:
                continue
            crop_image_mask = crop_image[:-9] + 'mask' + crop_image[-4:]
            f.write(crop_image + ' ' + crop_image_mask + '\n')
    if save_negative:
        i = 0
        for crop_image in negative_list:
            i += 1
            if np.mod(i, negative_ds_rate) != 0:
                continue
            crop_image_mask = crop_image[:-9] + 'mask' + crop_image[-4:]
            f.write(crop_image + ' ' + crop_image_mask + '\n')
    print('done')



def divide_img_mask(imgdir,crop_images,list):
    for crop_image in crop_images:
        if crop_image.endswith('image.png'):
            crop_image = imgdir + crop_image
            # crop_image_mask = crop_image[:-9] + 'mask' + crop_image[-4:]
            list.append(crop_image)
            # masks.append(crop_image_mask)
    return list

def get_image_list(root_dir,in_tissue_all,out_tissue_all,negative_all):

    image_names = os.listdir(root_dir)

    for image_name in image_names:
        dir = root_dir + image_name+'/'
        in_tissue_crops = os.listdir(dir+'in_tissue/')
        in_tissue_all = divide_img_mask(dir+'in_tissue/',in_tissue_crops,in_tissue_all)

        out_tissue_crops = os.listdir(dir + 'out_tissue/')
        out_tissue_all = divide_img_mask(dir + 'out_tissue/', out_tissue_crops,out_tissue_all)

        if os.path.exists(dir+'negative/'):
            negative_crops = os.listdir(dir+'negative/')
            negative_all = divide_img_mask(dir + 'negative/', negative_crops,negative_all)
    return in_tissue_all,out_tissue_all,negative_all

# ------------------------------------------
#                crop data loading
# ------------------------------------------

# image_root = '/home/huifang/workspace/data/fiducial_crop_w2_aug6_iter4/'
# save_file = '/home/huifang/workspace/data/imagelists/fiducial_width'+ str(circle_width)+'_with_0.125negative_downsample.txt'
#
# in_tissue_all = []
# out_tissue_all = []
# negative_all = []
# dataset_names = os.listdir(image_root)
# for dataset_name in dataset_names:
#     dataset_path = image_root + dataset_name + '/'
#     in_tissue_all,out_tissue_all,negative_all = get_image_list(dataset_path,in_tissue_all,out_tissue_all,negative_all)
#
# downsample_base = np.min([len(in_tissue_all),len(out_tissue_all),len(negative_all)])
#
# f = open(save_file,'w')
# write_image_list_to_file(f,in_tissue_all, out_tissue_all,negative_all, downsample_base=downsample_base)


root_path = '/home/huifang/workspace/data/fiducial_train/'
save_file = '/home/huifang/workspace/data/imagelists/fiducial_tissue_human.txt'
f = open(save_file, 'w')

dataset_names = os.listdir(root_path)
for dataset_name in dataset_names:
    print('-------------'+dataset_name+'-------------')
    image_names= os.listdir(root_path+dataset_name)
    for image_name in image_names:
        img_path = root_path+dataset_name+'/' + image_name
        if os.path.exists( img_path +'/tissue_hires_image.png'):
            tissue_image = img_path +'/tissue_hires_image.png'
        elif os.path.exists( img_path +'/spatial/tissue_hires_image.png'):
            tissue_image = img_path + '/spatial/tissue_hires_image.png'
        else:
            continue
        tissue_mask = root_path+dataset_name+'/' + image_name + '/masks/human_in_loop_mask_w2.png'
        f.write(tissue_image + ' ' + tissue_mask + '\n')
f.close()
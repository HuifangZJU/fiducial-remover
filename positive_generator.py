from __future__ import division
import os.path
import random
import matplotlib.pyplot as plt
import itertools
from hough_utils import *
from fiducial_utils import read_tissue_image as read_image

crop_size = 16
augsize =6
circle_width = 2

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

        crop_image = image[yc - crop_size:yc + crop_size, xc - crop_size:xc + crop_size, :]
        crop_mask = mask[yc - crop_size:yc + crop_size, xc - crop_size:xc + crop_size]
        # f,a = plt.subplots(1,2)
        # a[0].imshow(crop_image)
        # a[1].imshow(crop_mask)
        # plt.show()
        save_image(crop_image, path + str(yc) + '_' + str(xc) + '_'+ str(i)+'_image.png',format="RGB")
        save_image(crop_mask, path + str(yc) + '_' + str(xc) + '_'+ str(i)+'_mask.png', format="L")


# ------------------------------------------
#                data loading
# ------------------------------------------
negative_coordinates = './negative_coordinates.txt'
fn = open(negative_coordinates,'r')
negative_limits = fn.readlines()
for i in range(len(negative_limits)):
    negative_limits[i] = negative_limits[i].rstrip('\n')
    negative_limits[i] = negative_limits[i].split(' ')[1:5]
    negative_limits[i]=[int(x) for x in negative_limits[i]]


source_images = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
f = open(source_images, 'r')
fiducial_images = f.readlines()
in_tissue_list = []
out_tissue_list = []
negative_list = []
for i in range(0,len(fiducial_images)):
    image_name = fiducial_images[i].split(' ')[0]
    group_id = fiducial_images[i].split(' ')[-1]
    # print(image_name+'...')
    image_name = image_name.rstrip('\n')
    circles = np.load(image_name.split('.')[0] + '.npy')
    in_tissue = np.where(circles[:,-1]==1)
    out_tissue = np.where(circles[:,-1]==0)

    negative_xmin,negative_ymin,negative_xmax,negative_ymax = negative_limits[i]
    # mask_name = image_name.split('.')[0] + '_mask_width2.png'
    # mask_name = image_name.split('.')[0] + '_mask_solid.png'
    for id in in_tissue[0]:
        circle = circles[id, :]
        line = image_name + ' ' + str(circle[0]) + ' ' + str(circle[1]) + ' ' + group_id
        in_tissue_list.append(line)
        #keep negative the same number as in_tissue case
        xc = random.randint(negative_xmin,negative_xmax)
        yc = random.randint(negative_ymin,negative_ymax)
        line = image_name + ' ' + str(xc) + ' ' + str(yc) + ' ' + group_id
        negative_list.append(line)


    for id in out_tissue[0]:
        circle = circles[id, :]
        line = image_name + ' ' + str(circle[0]) + ' ' + str(circle[1]) + ' ' + group_id
        out_tissue_list.append(line)



target_image_path = '/home/huifang/workspace/data/imagelists/st_trainable_circles_downsample_with_negative_final.txt'
f = open(target_image_path,'w')
for line in in_tissue_list:
    f.write(f"{line}")
for i in range(0, len(out_tissue_list), 10):
    f.write(f"{out_tissue_list[i]}")
for line in negative_list:
    f.write(f"{line}")
print('done')


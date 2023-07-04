import os

import numpy as np
from matplotlib import pyplot as plt
import cv2

import fiducial_utils
from hough_utils import *
from fiducial_utils import *
from skimage.draw import disk


SAVE_ROOT = '/media/huifang/data/fiducial/annotation/'

def get_annotated_circles(annotation_path):
    in_tissue_path = os.path.join(annotation_path, 'in_tissue')
    in_tissue_circle = [circle for circle in os.listdir(in_tissue_path) if circle.endswith('image.png')]
    in_circle_meta = [[int(u), int(v), int(r), 1] for circle in in_tissue_circle for v, u, r, _ in [circle.split('_')]]

    out_tissue_path = os.path.join(annotation_path, 'auto')
    out_tissue_circle = [circle for circle in os.listdir(out_tissue_path) if circle.endswith('image.png')]
    out_circle_meta = [[int(u), int(v), int(r), 0] for circle in out_tissue_circle for v, u, r, _ in [circle.split('_')]]

    return in_circle_meta, out_circle_meta

def plot_circles_in_image(image,in_tissue_circles,out_tissue_circles, width):

    for circle in in_tissue_circles:
        cv2.circle(image, (circle[0],circle[1]), circle[2]+2,[255,0,0], width)

    for circle in out_tissue_circles:

        cv2.circle(image, (circle[0],circle[1]), circle[2]+2, [0, 255, 0], width)
    return image

def show_grids(image, cnt):
    h = image.shape[0]
    w = image.shape[1]
    plt.imshow(image)

    h_step = int(h / cnt)
    w_step = int(w / cnt)

    for i in range(0, cnt + 1):
        y = [i * h_step, i * h_step]
        x = [0, w]
        plt.plot(x, y, color="green", linewidth=2)
    # y = [h,h]
    # x= [0,w]
    # plt.plot(x, y, color="green", linewidth=3)

    for i in range(0, cnt + 1):
        y = [0, h]
        x = [i * w_step, i * w_step]
        plt.plot(x, y, color="green", linewidth=2)
    # y = [0, h]
    # x = [w, w]
    # plt.plot(x, y, color="green", linewidth=3)
    plt.show()




def annotate_patches(image_size, step, circles):
    w,h = image_size
    num_patches_w = w // step
    num_patches_h = h // step

    annotation = np.zeros((num_patches_w, num_patches_h), dtype=int)
    for i in range(num_patches_w):
        for j in range(num_patches_h):
            patch_x = i * step
            patch_y = j * step
            patch_rect = (patch_x, patch_y, step, step)

            for circle in circles:
                circle_x, circle_y, circle_radius = circle[:3]
                circle_radius = circle_radius+4
                circle_rect = (circle_y - circle_radius,circle_x - circle_radius,  2 * circle_radius, 2 * circle_radius)


                if rectangles_intersect(patch_rect, circle_rect):
                    annotation[i, j] = 1
                    # plt.imshow(image[patch_x:patch_x + step, patch_y:patch_y + step])
                    # print(annotation[i, j])
                    # plt.show()
                    break


    return annotation

def get_image_mask_from_annotation(image_size,annotation,step):
    image_mask = np.zeros(image_size)

    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            patch_x = i * step
            patch_y = j * step
            image_mask[patch_x:patch_x + step, patch_y:patch_y + step] = annotation[i, j]
    return image_mask

def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)



def get_annotation_path(imagepath):
    dataset = imagepath.split('/')[6]
    index = imagepath.split('/')[7]
    index = index.split('.')
    index = index[0]
    data_path = SAVE_ROOT + dataset + '_' + index
    return data_path


source_images = '/home/huifang/workspace/data/imagelists/st_image_trainable_fiducial.txt'
f = open(source_images, 'r')
fiducial_images = f.readlines()
SAVE_FILE = False
if SAVE_FILE:
    imagelist_path = '/home/huifang/workspace/data/imagelists/fiducial_auto_width2.txt'
    f_list = open(imagelist_path,'w')

width = 2
patch_size = 32
badfile=[109,113,114,116,119,129,131,132,136,137,138,144,152,153,154,160,161,162,163,164,165]
temp_image_list = '/home/huifang/workspace/data/imagelists/st_image_trainable_temp_fiducial.txt'
f_temp = open(temp_image_list,'w')
for i in range(0,len(fiducial_images)):
    if i in badfile:
        continue
    image_name = fiducial_images[i]
    f_temp.write(image_name)
    #
    # print(str(len(fiducial_images))+'---'+str(i))
    # image = plt.imread(image_name[:-1])
    # # show_grids(image,64)
    # annotation_path = get_annotation_path(image_name)
    # in_tissue_circles, out_tissue_circles = get_annotated_circles(annotation_path)
    #
    # circles = in_tissue_circles + out_tissue_circles
    #
    # patches = annotate_patches(image.shape[:2], patch_size,circles)
    # annotation_image = get_image_mask_from_annotation(image.shape[:2], patches, patch_size)
    # # image = plot_circles_in_image(image,in_tissue_circles,out_tissue_circles,width)
    # plt.imshow(image)
    # plt.imshow(annotation_image, cmap='binary', alpha=0.5)
    # plt.show()


    # show_circles_in_image(image,in_tissue_circles,out_tissue_circles,width)

    # if SAVE_FILE:
    #     save_path = annotation_path + '/masks/'
    #     image_mask = np.zeros(image.shape[:2])
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     save_name = save_path+'auto_width_2'
    #     save_mask_to_file(image_mask, circles, save_name)
    #     f_list.write(image_name[:-1]+' ' + save_name + '.png' + '\n')
print('done')
if SAVE_FILE:
    f_list.close()







#
#
# f = open(save_file,'w')
# image_paths = os.listdir(image_root_path)
# for image_path in image_paths:
#     temp_path = image_root_path+image_path+'/positive/'
#     images = os.listdir(temp_path)
#     write_to_file_list=[]
#     for image_name in images:
#         if image_name.endswith('mask.png'):
#             crop_image_mask = temp_path+image_name
#             crop_image = temp_path+image_name[:-9]+image_name[-4:]
#             # write_to_file_list.append(crop_image+' '+crop_image_mask)
#             f.write(crop_image+' '+crop_image_mask + '\n')
#     print(image_path + '  done.')
# print('all done')

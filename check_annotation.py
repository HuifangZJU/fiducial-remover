import os

import numpy as np
from matplotlib import pyplot as plt
import cv2

import fiducial_utils
from hough_utils import *
from fiducial_utils import *



root_path = '/home/huifang/workspace/data/fiducial_train/'
dataset_names = os.listdir(root_path)
annotation_path = '/home/huifang/workspace/data/fiducial_anno/'
width = -1

for dataset_name in dataset_names:
    print('-------------' + dataset_name + '-------------')
    image_names = os.listdir(root_path + dataset_name)
    for image_name in image_names:
        print(image_name+'...')
        image = fiducial_utils.read_tissue_image(root_path+dataset_name+'/'+image_name)
        image_mask = np.zeros(image.shape[:2])

        annotations = os.listdir(annotation_path+image_name)
        in_tissue_path = annotation_path+image_name+'/'+'in_tissue'
        in_tissue_circle = os.listdir(in_tissue_path)
        circles=[]
        for circle in in_tissue_circle:
            if circle.endswith('image.png'):
                v,u,r,_ =circle.split('_')
                cv2.circle(image, (int(u),int(v)), int(r)+2,[255,0,0], width)
                cv2.circle(image_mask, (int(u), int(v)), int(r)+2, 1, width)
                # cv2.circle(image_mask, (int(u), int(v)), int(r)-5, 1, 1)
                # for i in range(3):
                #     cv2.circle(image_mask, (int(u),int(v)), int(r)-3*i, 1, width)
                circles.append([int(u),int(v),int(r),1])

        out_tissue_path = annotation_path +image_name+'/'+'out_tissue'
        out_tissue_circle = os.listdir(out_tissue_path)
        for circle in out_tissue_circle:
            if circle.endswith('image.png'):
                v, u, r, _ = circle.split('_')
                cv2.circle(image, (int(u), int(v)), int(r)+2, [0, 255, 0], width)
                cv2.circle(image_mask, (int(u), int(v)),int(r)+2, 1, width)
                # cv2.circle(image_mask, (int(u), int(v)), int(r) - 5, 1, 1)
                # for i in range(3):
                #     cv2.circle(image_mask, (int(u), int(v)), int(r) -3*i, 1, width)
                circles.append([int(u), int(v), int(r),0])
        # plt.imshow(image,cmap='gray')
        # plt.show()
        save_path = root_path + dataset_name + '/' + image_name + '/masks/'

        save_mask_to_file(image_mask,circles,save_path+'human_in_loop_mask_solid_r2')
        # print('done')
        # test = input()







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

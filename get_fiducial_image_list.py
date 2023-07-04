from __future__ import division
import os

# # -------------------------------------------------------
# #         split w/o aligned fiducial
# # -------------------------------------------------------
# root_path = '/media/huifang/data/fiducial/data/'
# save_file_aligned = '/home/huifang/workspace/data/imagelists/st_image_with_aligned_fiducial.txt'
# save_file_no_aligned = '/home/huifang/workspace/data/imagelists/st_image_no_aligned_fiducial.txt'
# f_aligned = open(save_file_aligned, 'w')
# f_no_aligned = open(save_file_no_aligned, 'w')
#
# dataset_names = os.listdir(root_path)
# for dataset_name in dataset_names:
#     print('-------------' + dataset_name + '-------------')
#     image_names = os.listdir(root_path + dataset_name)
#     for image_name in image_names:
#         img_path = root_path + dataset_name + '/' + image_name
#         if '.' in image_name:
#             pass
#             f_no_aligned.write(img_path + '\n')
#         else:
#             image_sub_names = os.listdir(root_path + dataset_name + '/' + image_name)
#             for sub_name in image_sub_names:
#                 if 'tissue_hires_image' in sub_name:
#                     image_sub_path = img_path+'/' + sub_name
#                 if 'aligned_fiducials' in sub_name:
#                     image_aligned_sub_path = img_path + '/' + sub_name
#             f_aligned.write(image_sub_path + ' ')
#             f_aligned.write(image_aligned_sub_path + '\n')
# f_no_aligned.close()
# f_aligned.close()
# print('done')


# -------------------------------------------------------
#         split trainable images and hard images
# -------------------------------------------------------
import matplotlib.pyplot as plt

hard_index =[19, 20, 24, 44, 48, 49, 50, 51]
abandoned = [80, 81]
save_file_aligned = '/home/huifang/workspace/data/imagelists/st_image_with_aligned_fiducial.txt'
save_file_no_aligned = '/home/huifang/workspace/data/imagelists/st_image_no_aligned_fiducial.txt'
f_aligned = open(save_file_aligned, 'r')
aligned_images = f_aligned.readlines()
f_no_aligned = open(save_file_no_aligned, 'r')
no_aligned_images = f_no_aligned.readlines()
f_no_aligned.close()
f_aligned.close()

trainable_file = '/home/huifang/workspace/data/imagelists/st_image_trainable_fiducial.txt'
hard_file = '/home/huifang/workspace/data/imagelists/st_image_hard_fiducial.txt'
f_train = open(trainable_file, 'w')
f_hard = open(hard_file, 'w')

for image_path in aligned_images:
    image_path = image_path.split(' ')
    image_path = image_path[0]
    f_train.write(image_path + '\n')

for i in range(len(no_aligned_images)):
    if i in abandoned:
        continue
    image_path = no_aligned_images[i]
    if i in hard_index:
        f_hard.write(image_path)
    else:
        f_train.write(image_path)
f_train.close()
f_hard.close()

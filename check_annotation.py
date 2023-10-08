import os

import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.signal import convolve2d
import fiducial_utils
from hough_utils import *
from fiducial_utils import *
from skimage.draw import disk
import math
import random
# SAVE_ROOT = '/media/huifang/data/fiducial/annotation/'
SAVE_ROOT = '/media/huifang/data/fiducial/Fiducial_colab/Fiducial_colab/'

def get_annotated_circles(annotation_path):
    in_tissue_path = os.path.join(annotation_path, 'in_tissue')
    in_tissue_circle = [circle for circle in os.listdir(in_tissue_path) if circle.endswith('image.png')]
    in_circle_meta = [[int(u), int(v), int(r), 1] for circle in in_tissue_circle for v, u, r, _ in [circle.split('_')]]

    out_tissue_path = os.path.join(annotation_path, 'out_tissue')
    out_tissue_circle = [circle for circle in os.listdir(out_tissue_path) if circle.endswith('image.png')]
    out_circle_meta = [[int(u), int(v), int(r), 0] for circle in out_tissue_circle for v, u, r, _ in [circle.split('_')]]

    return in_circle_meta, out_circle_meta

def plot_circles_in_image(image,in_tissue_circles,out_tissue_circles, width):

    for circle in in_tissue_circles:
        cv2.circle(image, (circle[0],circle[1]), circle[2],[255,0,0], width)

    for circle in out_tissue_circles:

        cv2.circle(image, (circle[0],circle[1]), circle[2], [0, 255, 0], width)
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

def calculate_distance(point1, point2):
    """Calculate the distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def unique_pairs_below_threshold(circle_list, threshold):
    """Find unique pairs of circles with distance below a given threshold."""
    result = []
    for i in range(len(circle_list)):
        for j in range(i+1, len(circle_list)):
            distance = calculate_distance(circle_list[i][:2], circle_list[j][:2])
            if distance < threshold:
                result.append((str(circle_list[i][1])+'_'+str(circle_list[i][0])+'_'+str(circle_list[i][2]),
                               str(circle_list[j][1])+'_'+str(circle_list[j][0])+'_'+str(circle_list[j][2])))
    return result


def pairwise_distances(matrix):
    """Compute pairwise distances using matrix operations."""
    diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return distances


def remove_overlapping_circles(circle_list, threshold):
    """Remove circles until there are no pairs with distance below the threshold."""
    matrix = np.array([circle[:2] for circle in circle_list])

    while True:
        distances = pairwise_distances(matrix)
        # Set the diagonal to a large value to exclude self-comparison
        np.fill_diagonal(distances, float('inf'))

        # Check if there's any distance below the threshold
        i, j = np.where(distances < threshold)

        if len(i) == 0:
            break

        # Remove one random circle from a pair that violates the threshold
        if circle_list[i[0]][2]>circle_list[j[0]][2]:
            to_remove = j[0]
        elif circle_list[i[0]][2]<circle_list[j[0]][2]:
            to_remove = i[0]
        else:
            to_remove = random.choice([i[0], j[0]])
        matrix = np.delete(matrix, to_remove, axis=0)
        circle_list.pop(to_remove)

    return circle_list
def annotate_patches(image_size, patch_size, circles):
    w,h = image_size
    num_patches_w = w // patch_size
    num_patches_h = h // patch_size

    annotation = np.zeros((num_patches_w, num_patches_h), dtype=int)
    for i in range(num_patches_w):
        for j in range(num_patches_h):
            patch_x = i * patch_size
            patch_y = j * patch_size
            patch_rect = (patch_x, patch_y, patch_size, patch_size)

            for circle in circles:
                circle_x, circle_y, circle_radius = circle[:3]
                # circle_radius = circle_radius+4
                circle_rect = (circle_y - circle_radius,circle_x - circle_radius,  2 * circle_radius, 2 * circle_radius)


                if rectangles_intersect(patch_rect, circle_rect):
                    annotation[i, j] = 1
                    # plt.imshow(image[patch_x:patch_x + step, patch_y:patch_y + step])
                    # print(annotation[i, j])
                    # plt.show()
                    break
    # filling holes
    kernel = np.ones((3, 3))
    neighbor_count = convolve2d(annotation, kernel, mode='same')
    holes = np.logical_and(annotation == 0, neighbor_count > 4)
    while np.sum(holes)>0:
        annotation= np.logical_or(annotation,holes)
        neighbor_count = convolve2d(annotation, kernel, mode='same')
        holes = np.logical_and(annotation == 0, neighbor_count > 4)
    # neighbor_count = convolve2d(annotation, kernel, mode='same')
    return annotation


def annotate_continuous_patches(image_size, patch_size, circles):

    # Initialize the mask with zeros
    mask = np.zeros(image_size, dtype=np.uint8)

    half_patch = patch_size // 2
    for circle in circles:
        x, y, _, _ = circle

        # Calculate the top-left and bottom-right corners of the square patch
        top_left_x = max(0, int(x - half_patch))
        top_left_y = max(0, int(y - half_patch))
        bottom_right_x = min(image_size[1], int(x + half_patch))
        bottom_right_y = min(image_size[0], int(y + half_patch))

        # Set the square patch to 1 in the mask
        mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
    # filling holes
    # kernel = np.ones((9, 9), np.uint8)
    #
    # # Apply closing operation
    # closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #
    # # Detect border pixels in the original mask
    # border = cv2.filter2D(mask, -1, kernel)
    # border_mask = (border >= 3) & (border < 8)
    #
    # # Restore the border pixels in the closed mask
    # closed_mask[border_mask] = mask[border_mask]
    return mask

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
# badfile=[109,113,114,116,119,129,131,132,136,137,138,144,152,153,154,160,161,162,163,164,165]
# temp_image_list = '/home/huifang/workspace/data/imagelists/st_image_temp_trainable_fiducial.txt'
# f_temp = open(temp_image_list,'w')
for i in range(0,len(fiducial_images)):
    # if i in badfile:
    #     continue
    image_name = fiducial_images[i].split(' ')[0]
    image_name = image_name.rstrip('\n')
    # f_temp.write(image_name)

    print(str(len(fiducial_images))+'---'+str(i))
    image = plt.imread(image_name)
    # plt.imshow(image)
    # plt.show()
    # image = image*255
    # image = np.array(image,dtype=np.uint8)
    # file_name = f"{i}.png"
    # plt.imsave('./temp/'+file_name,image)
    # show_grids(image,64)

    annotation_path = get_annotation_path(image_name)
    if not os.path.exists(annotation_path):
        continue
    print(image_name)
    in_tissue_circles, out_tissue_circles = get_annotated_circles(annotation_path)
    # #
    if len(unique_pairs_below_threshold(in_tissue_circles, 10))>0:
        in_tissue_circles = remove_overlapping_circles(in_tissue_circles, 10)
    if len(unique_pairs_below_threshold(out_tissue_circles, 10))>0:
        out_tissue_circles = remove_overlapping_circles(out_tissue_circles, 10)
    circles = in_tissue_circles + out_tissue_circles

    # print(unique_pairs_below_threshold(in_tissue_circles, 10))
    # print(len(unique_pairs_below_threshold(out_tissue_circles, 10)))
    print(len(circles))

    # np.save(annotation_path+'/circles.npy',circles)

    #
    # patches = annotate_patches(image.shape[:2], patch_size,circles)
    # annotation_image = get_image_mask_from_annotation(image.shape[:2], patches, patch_size)
    annotation_image = annotate_continuous_patches(image.shape[:2], 32,circles)
    image = plot_circles_in_image(image,in_tissue_circles,out_tissue_circles,width)
    f,a = plt.subplots(1,2,figsize=(16, 8))
    # plt.figure(figsize=(8, 8))
    a[0].imshow(image)
    a[0].imshow(annotation_image, cmap='binary', alpha=0.5)
    a[1].imshow(annotation_image)
    plt.show()


    # if SAVE_FILE:
    #     save_path = annotation_path + '/masks/'
    #     image_mask = np.zeros(image.shape[:2])
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     save_name = save_path+'auto_width_2'
    #     save_mask_to_file(image_mask, circles, save_name)
    #     f_list.write(image_name[:-1]+' ' + save_name + '.png' + '\n')
print('done')
# if SAVE_FILE:
#     f_list.close()







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

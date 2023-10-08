import os

import numpy as np
from matplotlib import pyplot as plt


import fiducial_utils
from hough_utils import *
from fiducial_utils import *
from hough_utils import *
import fiducial_utils
from fiducial_utils import read_tissue_image as read_image
import cv2

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
            image_mask[patch_x:patch_x + step, patch_y:patch_y + step,:] = annotation[i, j,:]
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


def extract_integer(filename):
    return int(filename.split('.')[0].split('_')[0])


def divide_image_into_patches(image, patch_size, margin_size):
    image_height, image_width, _ = image.shape
    patches = []

    for y in range(0, image_height, patch_size):
        for x in range(0, image_width, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size].copy()
            patch_with_margin = cv2.copyMakeBorder(patch, margin_size, margin_size, margin_size, margin_size,
                                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
            patches.append(patch_with_margin)

    return patches






result_path = '/media/huifang/data/experiment/pix2pix/images/binary-square-alltrain-from-scratch/'
images = [img for img in os.listdir(result_path) if img.endswith('img.png')]
images =sorted(images, key=extract_integer)

gts = [img for img in os.listdir(result_path) if img.endswith('gt.png')]
gts =sorted(gts, key=extract_integer)

predictions = [img for img in os.listdir(result_path) if img.endswith('mask.png')]
predictions =sorted(predictions, key=extract_integer)

patch_size = 32
margin_size = 8



for i in range(len(images)-16,len(images)):
    image_name = images[i]
    gt_name = gts[i]
    prediction_name = predictions[i]

    image = plt.imread(result_path + image_name)
    gt = plt.imread(result_path + gt_name)
    pre = plt.imread(result_path + prediction_name)
    print(image.shape)
    print(pre.shape)
    print(patch_size)
    test = input()


    anno_gt = get_image_mask_from_annotation(image.shape, gt, patch_size)
    anno_pre = get_image_mask_from_annotation(image.shape, pre, patch_size)

    anno_pre = np.where(anno_pre > 0.2, 1.0, 0.0)
    output = image.copy()

    # image = image*anno_pre
    #
    # circles = run_circle_threhold(image, 11, circle_threshold=25)
    # for i in range(circles.shape[0]):
    #     cv2.circle(output, (circles[i, 0], circles[i, 1]), circles[i, 2], (0, 255, 0), 3)
    # plt.imshow(output)
    # plt.show()
    #


    # pre2 = denoise_tv_bregman(pre[:,:,0],weight=0.5)
    # pre2 = np.where(anno_pre>0.2,1,0)
    # plt.figure(1)
    # plt.imshow(anno_pre)
    # plt.figure(2)
    # plt.imshow(pre2)
    # plt.show()


    #
    f, a = plt.subplots(1, 2,figsize=(20, 15))
    a[0].imshow(output)
    a[0].imshow(anno_gt, cmap='binary', alpha=0.5)
    a[0].set_title("ground truth")

    a[1].imshow(output)
    a[1].imshow(anno_pre, cmap='binary', alpha=0.5)
    a[1].set_title("prediction")
    # plt.pause(0.3)
    # a[0].cla()
    # a[1].cla()
    plt.show()











# image_name = fiducial_images[i]
# # f_temp.write(image_name)
#
# print(str(len(fiducial_images))+'---'+str(i))
# image = plt.imread(image_name[:-1])
# # image = image*255
# # image = np.array(image,dtype=np.uint8)
# file_name = f"{i}.png"
# plt.imsave('./temp/'+file_name,image)

# show_grids(image,64)

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


# if SAVE_FILE:
#     save_path = annotation_path + '/masks/'
#     image_mask = np.zeros(image.shape[:2])
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     save_name = save_path+'auto_width_2'
#     save_mask_to_file(image_mask, circles, save_name)
#     f_list.write(image_name[:-1]+' ' + save_name + '.png' + '\n')

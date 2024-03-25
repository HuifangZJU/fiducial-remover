from __future__ import division
import os.path
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from hough_utils import *
import fiducial_utils
import seaborn as sns

IN_TISSUE = "1"
WRONG = "0"
OUT_TISSUE = "2"
format = 'manual'
# format = 'auto'
SAVE_ROOT = '/media/huifang/data/fiducial/annotation/'

#TODO: give a radius range, select the best one
he_radius_base = 9
fiducial_radius_base = 16
crop_size = 16


# y is the v direction
# x is the u direction
def save_annotation(crop_image,mask,label,dataset_path,x,y,r):
    if label == IN_TISSUE:
        save_path = dataset_path+ '/in_tissue/' + str(y) + '_' + str(x) + '_' + str(r)
        save_image(crop_image, save_path +'_image.png', format="RGB")
        save_image(mask, save_path + '_mask.png' , format="L")
    if label == OUT_TISSUE:
        save_path = dataset_path+ '/out_tissue/' + str(y) + '_' + str(x) + '_' + str(r)
        save_image(crop_image, save_path + '_image.png', format="RGB")
        save_image(mask, save_path + '_mask.png', format="L")
    if label == WRONG:
        save_path = dataset_path+ '/hard/' + str(y) + '_' + str(x) + '_' + str(r)
        save_image(crop_image, save_path + '_image.png', format="RGB")
        save_image(mask, save_path + '_mask.png', format="L")
    if label == 'auto':
        save_path = dataset_path + '/auto/' + str(y) + '_' + str(x) + '_' + str(r)
        save_image(crop_image, save_path + '_image.png', format="RGB")
        save_image(mask, save_path + '_mask.png', format="L")

def split_array(arr, cond):
  return arr[cond], arr[~cond]

# circl in local image coordinate
def get_mask(img_shape,circle):
    mask = np.zeros(img_shape)
    cv2.circle(mask, (circle[0], circle[1]), circle[2], 1, 1)
    return mask
def get_masked_image(image,circle):
    cv2.circle(image, (circle[0], circle[1]), circle[2], 1, 1)
    return image


def get_translated_circles(image,circles, circles_f,fiducialcenter_x, fiducialcenter_y, fiducial_scale):
    framecenter_x, framecenter_y, square_scale = get_square_paras(image.shape[1], image.shape[0], circles, image)
    if not square_scale:
        print('Can not localize fiducial frame!')
        return
    # fiducial_scale = fiducial_scale + 2 * fiducial_radius_base
    square_scale[0] = square_scale[0] + 4 * he_radius_base
    # square_scale[1] = square_scale[1] + 2 * he_radius_base
    if framecenter_x & framecenter_y:
        # normalization
        circles_f[:, 0] = circles_f[:, 0] - fiducialcenter_x
        circles_f[:, 1] = circles_f[:, 1] - fiducialcenter_y

        circles_f[:, 0] = circles_f[:, 0] * square_scale[0] / fiducial_scale + framecenter_x
        circles_f[:, 1] = circles_f[:, 1] * square_scale[1] / fiducial_scale + framecenter_y
    # circles_f[:, :2] = circles_f[:, :2] * square_scale / fiducial_scale
    return circles_f



def annotate_single_image(image,he_radius,dataset_path,xf,yf):
    #run circle detection
    crop_image = image[yf - crop_size:yf + crop_size, xf - crop_size:xf + crop_size, :]
    crop_circle,_ = run_circle_max(crop_image, radius=he_radius, max_n=1, step=2)

    xc = crop_circle[0] + xf - crop_size
    yc = crop_circle[1] + yf - crop_size
    rc = crop_circle[2]
    # mask = get_mask(crop_image.shape[0:2], crop_circle)
    crop_image_show = crop_image.copy()
    mask = get_masked_image(crop_image, crop_circle)
    # cv2.circle(crop_image_show, (crop_circle[0], crop_circle[1]), crop_circle[2], (1, 1, 1), 1)
    f,axarr = plt.subplots(1,3,figsize=(20,10))
    axarr[0].imshow(crop_image)
    axarr[1].imshow(crop_image_show)
    axarr[2].imshow(mask,cmap='gray')
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
    print("Save or not: ")
    label = input()
    save_annotation(crop_image_show,crop_image,label,dataset_path,xc,yc,rc)


def save_circle_with_position(image, circle,dataset_path):
    xc = circle[0]
    yc = circle[1]
    rc = circle[2]
    x0 = xc
    y0 = yc

    x_min, y_min, x_max, y_max = fiducial_utils.getCropCoor(x0, y0, crop_size, image.shape[1], image.shape[0])

    crop_image = image[y_min:y_max, x_min:x_max, :]
    crop_image_show = crop_image.copy()
    circle[0] = circle[0] - x0 + crop_size
    circle[1] = circle[1] - y0 + crop_size
    # mask = get_mask(crop_image.shape, circle[:])
    mask = get_masked_image(crop_image, circle[:])
    #

    # cv2.circle(crop_image_show, (circle[0], circle[1]), circle[2], (1, 1, 1), 1)
    # plt.imshow(crop_image_show)
    # plt.show()
    save_annotation(crop_image_show, crop_image, 'auto', dataset_path, xc, yc,rc)

def save_easy_circles(image, dataset_path, easy_indices,circles):
    if not easy_indices:
        for circle in circles:
            save_circle_with_position(image, circle, dataset_path)
        print('auto annotation successful, saved ' + str(len(circles)) + ' files.')
    else:
        for i in easy_indices:
            circle = circles[i, :]
            save_circle_with_position(image, circle, dataset_path)
        print('auto annotation successful, saved ' + str(len(easy_indices)) + ' files.')

# xc, yc, circle position in original image
# x0, y0, crop center

def run_self(imagepath):
    image_cv = cv2.imread(imagepath)
    circles = run_circle_threhold(image_cv, he_radius_base, circle_threshold=20,edgemethod='self', step=4)

    image = plt.imread(imagepath)
    # circles = run_circle_threhold(image, he_radius_base, circle_threshold=25, step=1)
    DEBUG = True
    if DEBUG:
        output = image.copy()
        for i in range(circles.shape[0]):
            cv2.circle(output, (circles[i, 0], circles[i, 1]), circles[i, 2], (0, 255, 0), 2)
        plt.imshow(output)
        plt.show()

    for i in range(0,circles.shape[0]):
        circles[i,2] = circles[i,2]-2
    # #if use the initial detection result
    easy_circles = circles
    indices, distance = find_nearest_points(easy_circles[:, :2], easy_circles[:, :2], is_same=True)
    # delete_index=[]
    # for i in range(easy_circles.shape[0]):
    #     if distance[i]>30:
    #         delete_index.append(i)
    # easy_circles = np.delete(easy_circles,delete_index,0)
    manual_positions = []
    return easy_circles, manual_positions, he_radius_base



def run_hough(image, imagepath, aligned_path,shrink):

    if aligned_path:
        print('Using user provided fiducials.')
        transposed_fiducial, scale = fiducial_utils.runCircle(aligned_path)
        he_radius = round(he_radius_base*scale*1.1)
        crop_size = 2 * he_radius
        print(he_radius)
        f_indices = np.arange(transposed_fiducial.shape[0])
    else:
        print('Using mouse fiducials.')
        he_radius = he_radius_base
        crop_size = 2 * he_radius

        circles = run_circle_threhold(image, he_radius, circle_threshold=18, step=3)
        output = image.copy()
        for i in range(circles.shape[0]):
            cv2.circle(output, (circles[i, 0], circles[i, 1]), circles[i, 2], (0, 255, 0), 2)
        plt.imshow(output)
        plt.show()

        circles_copy = circles.copy()
        circles_f_, fiducialcenter_x, fiducialcenter_y, fiducial_scale = fiducial_utils.mouse_para()
        circles_f = circles_f_.copy()


        #scale translation of fiducial circles
        if shrink > 0:
            print('Use user provided scale ' + str(shrink) + '.')
            # horizon
            circles_f[:, 0] = circles_f[:, 0] * shrink
            # vertical
            circles_f[:, 1] = circles_f[:, 1] * shrink
        else:
            circles_f = get_translated_circles(image, circles,circles_f, fiducialcenter_x, fiducialcenter_y, fiducial_scale)

        circles_f_copy = circles_f.copy()
        transposed_fiducial = get_transposed_fiducials(circles, circles_f, iter=200)
        c_indices, distance = find_nearest_points(circles[:, :2], transposed_fiducial[:, :2])
        f_indices = np.arange(c_indices.shape[0])
    DEBUG=True
    if DEBUG:
        # output = image.copy()
        # for i in range(circles.shape[0]):
        #     cv2.circle(output, (circles[i, 0], circles[i, 1]), circles[i, 2], (0, 255, 0), 2)
        # plt.imshow(output)
        # f, axarr = plt.subplots(1, 3)
        # plt.setp(axarr, xticks=[], yticks=[])
        # axarr[0].imshow(image)
        # axarr[1].imshow(output)
        # axarr[2].imshow(image)
        # axarr[2].scatter(circles[:, 0], circles[:, 1], marker='.', color="red", s=1)
        plt.show()

        f2, axarr2 = plt.subplots(2,2,figsize=(30,15))
        plt.setp(axarr2, xticks=[], yticks=[])
        axarr2[0,0].scatter(circles_copy[:, 0], circles_copy[:, 1])
        axarr2[0,0].scatter(circles_f_[:, 0], circles_f_[:, 1])
        axarr2[0,0].axis('equal')

        axarr2[0, 1].scatter(circles[:, 0], circles[:, 1])
        axarr2[0, 1].scatter(circles_f_copy[:, 0], circles_f_copy[:, 1])
        axarr2[0, 1].axis('equal')

        axarr2[1,0].scatter(circles[:, 0], circles[:, 1])
        axarr2[1,0].scatter(transposed_fiducial[:, 0], transposed_fiducial[:, 1])
        axarr2[1,0].axis('equal')
        output_fiducial = image.copy()
        for i in range(transposed_fiducial.shape[0]):
            cv2.circle(output_fiducial, (transposed_fiducial[i, 0], transposed_fiducial[i, 1]), round(fiducial_radius_base), (0, 255, 0), 2)
        axarr2[1,1].imshow(output_fiducial)
        plt.show()
    # -------------------------------------------------------
    #         begin annotation
    # -------------------------------------------------------
    easy_circles_c, _ = split_array(c_indices,distance<he_radius)
    easy_circles_f, hard_circles_f = split_array(f_indices,distance<he_radius)
    # save_easy_circles(image, dataset_path, easy_circles_c, circles)
    ###########   automatically save easy circles  ********
    # redect circles
    #     for id in hard_circles_f:
    crop_circles=[]
    likelihoods=[]
    for id in hard_circles_f:
        xf = transposed_fiducial[id, 0]
        yf = transposed_fiducial[id, 1]
        x_min, y_min, x_max, y_max = fiducial_utils.getCropCoor(xf,yf,crop_size-4,image.shape[1],image.shape[0])
        crop_image = image[y_min:y_max, x_min:x_max, :]

        crop_circle, likelihood = run_circle_max(crop_image, radius=he_radius, max_n=1, step=2)
        crop_circles.append(crop_circle)
        likelihoods.append(likelihood)

    easy_positions_f = []
    easy_circles = []
    hard_positions_f = []
    manual_positions=[]

    for i in range(0,len(easy_circles_f)):
        f_id=easy_circles_f[i]
        easy_positions_f.append(transposed_fiducial[f_id,:2])
        c_id = easy_circles_c[i]
        circle = circles[c_id]
        easy_circles.append(circle)
    for i in range(0,len(hard_circles_f)):
        f_id = hard_circles_f[i]
        hard_positions_f.append(transposed_fiducial[f_id,:2])
    for i in range(0,len(hard_positions_f)):
        id = hard_circles_f[i]
        crop_circle = crop_circles[i]
        likelihood = likelihoods[i]
        if likelihood < 5:
            manual_positions.append(transposed_fiducial[id,:2])
        else:
            easy_positions_f.append(transposed_fiducial[id,:2])
            crop_circle[0] = crop_circle[0] + transposed_fiducial[id, 0] - crop_size
            crop_circle[1] = crop_circle[1] + transposed_fiducial[id, 1] - crop_size
            easy_circles.append(crop_circle)
    return easy_circles, manual_positions, he_radius




def run(imagepath,aligned_path,save_file=True,shrink=0.0):


    image = plt.imread(imagepath)


    # easy_circles, manual_positions, he_radius = run_self(imagepath)
    easy_circles, manual_positions, he_radius = run_hough(image, imagepath, aligned_path,shrink)
    # print(str(len(easy_circles))+' auto detected circles, '+str(len(manual_positions)) + ' need manual annotation.')

    #easy circles visuailization
    image_show = image.copy()
    for circle in easy_circles:
        cv2.circle(image_show, (circle[0], circle[1]), circle[2], 1, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(image_show)
    plt.show()

    if save_file:
        dataset = imagepath.split('/')[6]
        index = imagepath.split('/')[7]
        index = index.split('.')
        index = index[0]
        dataset_path = SAVE_ROOT + dataset + '_' + index
        os.makedirs(dataset_path + '/in_tissue', exist_ok=True)
        os.makedirs(dataset_path + '/out_tissue', exist_ok=True)
        os.makedirs(dataset_path + '/hard', exist_ok=True)
        os.makedirs(dataset_path + '/auto', exist_ok=True)
        save_easy_circles(image, dataset_path, [], easy_circles)
        i=0
        for fiducial in manual_positions:
            print(str(len(manual_positions))+'---'+str(i))
            xf = fiducial[0]
            yf = fiducial[1]
            annotate_single_image(image, he_radius,dataset_path, xf, yf)
            i += 1
        print( 'annotation successful, saved ' + str(len(easy_circles)+len(manual_positions)) + ' files.')
        print('done')
# ------------------------------------------
#                load imagelist
# ------------------------------------------
f_image = open('/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_no_aligned_fiducial.txt')
image_list = f_image.readlines()
#19 20 24 44 48 49 50 51 super hard

for i in range(0,len(image_list)):
    print('--- '+str(i)+' ---')
    image = image_list[i]
    image = image.split(' ')
    print(image[0])
    if len(image)==1:
        run(image[0][:-1],[], save_file=False,shrink=1.825)

    else:

    # try:
        run(image[0],image[1][:-1],save_file=False)

    # except:
    #     print("auto annotation failed")
    # continue

print('done')


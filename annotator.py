from __future__ import division
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np

from hough_utils import *
import get_template_para

IN_TISSUE = "1"
HARD = "0"
OUT_TISSUE = "2"
format = 'manual'
# format = 'auto'
SAVE_ROOT = '/home/huifang/workspace/data/fiducial_anno2/'

#TODO: give a radius range, select the best one
he_radius = 9
fiducial_radius = 15
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
    if label == HARD:
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


def annotate_single_image(image,dataset_path,xf,yf):
    #run circle detection
    crop_image = image[yf - crop_size:yf + crop_size, xf - crop_size:xf + crop_size, :]
    crop_circle,_ = run_circle_max(crop_image, radius=he_radius, max_n=1, step=2)

    xc = crop_circle[0] + xf - crop_size
    yc = crop_circle[1] + yf - crop_size
    rc = crop_circle[2]
    mask = get_mask(crop_image.shape[0:2], crop_circle)

    crop_image_show = crop_image.copy()
    cv2.circle(crop_image_show, (crop_circle[0], crop_circle[1]), crop_circle[2], (1, 1, 1), 1)
    f,axarr = plt.subplots(1,3)
    axarr[0].imshow(crop_image)
    axarr[1].imshow(crop_image_show)
    axarr[2].imshow(mask,cmap='gray')
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
    print("Save or not: ")
    label = input()
    save_annotation(crop_image,mask,label,dataset_path,xc,yc,rc)


def save_circle_with_position(image, circle,dataset_path):
    xc = circle[0]
    yc = circle[1]
    rc = circle[2]
    x0 = xc
    y0 = yc
    crop_image = image[y0 - crop_size:y0 + crop_size, x0 - crop_size:x0 + crop_size, :]
    circle[0] = circle[0] - x0 + crop_size
    circle[1] = circle[1] - y0 + crop_size
    mask = get_mask(crop_image.shape, circle[:])
    #
    # crop_image_show = crop_image.copy()
    # cv2.circle(crop_image_show, (circle[0], circle[1]), circle[2], (1, 1, 1), 1)
    # plt.imshow(crop_image_show)
    # plt.show()
    save_annotation(crop_image, mask, 'auto', dataset_path, xc, yc,rc)

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
def run(imagepath,dataset_name,save_file=True,shrink=0.0):


    dataset_path = SAVE_ROOT+dataset_name
    if save_file:
        os.makedirs(dataset_path+'/in_tissue', exist_ok=True)
        os.makedirs(dataset_path+'/out_tissue', exist_ok=True)
        os.makedirs(dataset_path+'/hard', exist_ok=True)
        os.makedirs(dataset_path + '/auto', exist_ok=True)

    try:
        image = plt.imread(imagepath + dataset_name +'/tissue_hires_image.png')
    except:
        try:
            image = plt.imread(imagepath + dataset_name + '/spatial/tissue_hires_image.png')
        except:
            print("Cannot find tissue_hires_image for dataset " + dataset_name + " !")
            return


    DEBUG=False
    circles = run_circle_threhold(image,he_radius,circle_threshold=30)
    if DEBUG:
        output = image.copy()
        for i in range(circles.shape[0]):
            cv2.circle(output, (circles[i,0], circles[i,1]), circles[i,2], (0, 255, 0), 2)
        f, axarr = plt.subplots(1,3)
        plt.setp(axarr, xticks=[], yticks=[])
        axarr[0].imshow(image)
        axarr[1].imshow(output)
        axarr[2].imshow(image)
        axarr[2].scatter(circles[:,0],circles[:,1],marker='.',color="red",s=1)
        plt.show()

    if os.path.exists(imagepath + dataset_name+'/aligned_fiducials.jpg') or os.path.exists(imagepath + dataset_name+'/spatial/aligned_fiducials.jpg'):
        print('Using user provided fiducials.')
        try:
            transposed_fiducial = get_template_para.runCircle(imagepath + dataset_name+'/aligned_fiducials.jpg',fiducial_radius)
        except:
            transposed_fiducial = get_template_para.runCircle(imagepath + dataset_name+ '/spatial/aligned_fiducials.jpg',fiducial_radius)
    else:
        print('Using mouse fiducials.')
        circles_f, fiducialcenter_x, fiducialcenter_y, fiducial_scale = get_template_para.mouse_para()
        if shrink>0:
            print('Use user provided scale ' + str(shrink)+'.')
            circles_f[:, :2] = circles_f[:, :2] * shrink
        else:
            framecenter_x,framecenter_y,square_scale = get_square_paras(image.shape[0],image.shape[1],circles,image)
            if not square_scale:
                print('Can not localize fiducial frame!')
                return
            if framecenter_x:
                circles_f[:,0] = circles_f[:,0] - fiducialcenter_x + framecenter_x
            if framecenter_y:
                circles_f[:,1] = circles_f[:,1] - fiducialcenter_y + framecenter_y
            circles_f[:,:2] = circles_f[:,:2]*square_scale/fiducial_scale

        # find alignment to fiducial based on ICP
        transposed_fiducial = get_transposed_fiducials(circles, circles_f, iter=20)

    if not DEBUG:
        f2, axarr2 = plt.subplots(1,3)
        plt.setp(axarr2, xticks=[], yticks=[])
        axarr2[0].scatter(circles[:, 0], circles[:, 1])
        axarr2[0].axis('equal')
        axarr2[1].scatter(circles[:, 0], circles[:, 1])
        axarr2[1].scatter(transposed_fiducial[:, 0], transposed_fiducial[:, 1])
        axarr2[1].axis('equal')
        output_fiducial = image.copy()
        for i in range(transposed_fiducial.shape[0]):
            cv2.circle(output_fiducial, (transposed_fiducial[i, 0], transposed_fiducial[i, 1]), fiducial_radius, (0, 255, 0), 2)
        axarr2[2].imshow(output_fiducial)
        plt.show()
    # -------------------------------------------------------
    #         begin annotation
    # -------------------------------------------------------
    c_indices, distance = find_nearest_points(circles[:,:2],transposed_fiducial[:,:2])
    f_indices = np.arange(c_indices.shape[0])
    # easy_circles_c, _ = split_array(c_indices,distance<he_radius)
    # _, hard_circles_f = split_array(f_indices,distance<he_radius)
    # save_easy_circles(image, dataset_path, easy_circles_c, circles)
    ###########   automatically save easy circles  ********
    # redect circles
    #     for id in hard_circles_f:
    crop_circles=[]
    likelihoods=[]
    for id in f_indices:
        xf = transposed_fiducial[id, 0]
        yf = transposed_fiducial[id, 1]
        crop_image = image[yf - crop_size:yf + crop_size, xf - crop_size:xf + crop_size, :]
        crop_circle, likelihood = run_circle_max(crop_image, radius=he_radius, max_n=1, step=2)
        crop_circles.append(crop_circle)
        likelihoods.append(likelihood)

    easy_positions_f = []
    easy_circles_in_f = []
    hard_positions_f = []
    for id in f_indices:
        crop_circle = crop_circles[id]
        likelihood = likelihoods[id]
        if likelihood < 20:
            hard_positions_f.append(transposed_fiducial[id,:2])
        else:
            easy_positions_f.append(transposed_fiducial[id,:2])
            crop_circle[0] = crop_circle[0] + transposed_fiducial[id, 0] - crop_size
            crop_circle[1] = crop_circle[1] + transposed_fiducial[id, 1] - crop_size
            easy_circles_in_f.append(crop_circle)
    print(str(len(easy_positions_f))+' auto detected circles, '+str(len(hard_positions_f)) + ' need manual annotation.')
    #easy circles visuailization
    image_show = image.copy()
    for circle,position in zip(easy_circles_in_f,easy_positions_f):
        cv2.circle(image_show, (circle[0], circle[1]), circle[2], 1, 1)
    plt.imshow(image_show)
    plt.show()
    if save_file:
        save_easy_circles(image, dataset_path, [], easy_circles_in_f)
        i=0
        for fiducial in hard_positions_f:
            print(str(len(hard_positions_f))+'---'+str(i))
            xf = fiducial[0]
            yf = fiducial[1]
            annotate_single_image(image, dataset_path, xf, yf)
            i += 1
        print( 'annotation successful, saved ' + str(len(easy_circles_in_f)+len(hard_positions_f)) + ' files.')
        print('done')
# ------------------------------------------
#                data loading
# ------------------------------------------
# imagepath = '/home/huifang/workspace/data/humanpilot/'
# imagepath = '/home/huifang/workspace/data/mouse/'
imagepath = '/home/huifang/workspace/data/fiducial_train/humanpilot/'
dataset_names = os.listdir(imagepath)
# dataset_name = 'spatial3'
for dataset_name in dataset_names:
    # dataset_name = 'spatial1'
    print('---'+dataset_name)
    run(imagepath, dataset_name,shrink=0.87, save_file=True)
    print('done')
    # test = input()
# try:
#     run(imagepath, dataset_name,save_file=False)
# except:
#     print("auto annotation failed")
    # continue



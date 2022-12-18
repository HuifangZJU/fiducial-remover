from __future__ import division
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np

from hough_utils import *
import get_template_para
from fiducial_utils import read_tissue_image as read_image

H_RADIUS = 9
crop_size = 16

# xc, yc, circle position in original image
# x0, y0, crop center
def run(imagepath,dataset_name,circle_likelihood=None,shrink=0.87):

    image = read_image(imagepath + dataset_name)
    circles = run_circle_threhold(image,he_radius,circle_threshold=30)

    if os.path.exists(imagepath + dataset_name+'/aligned_fiducials.jpg') or os.path.exists(imagepath + dataset_name+'/spatial/aligned_fiducials.jpg'):
        print('Using user provided fiducials.')
        try:
            transposed_fiducial = get_template_para.runCircle(imagepath + dataset_name+'/aligned_fiducials.jpg')
        except:
            transposed_fiducial = get_template_para.runCircle(imagepath + dataset_name+ '/spatial/aligned_fiducials.jpg')
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

    f_indices = np.arange(transposed_fiducial.shape[0])
    crop_circles=[]
    likelihoods=[]
    for id in f_indices:
        xf = transposed_fiducial[id, 0]
        yf = transposed_fiducial[id, 1]
        crop_image = image[yf - crop_size:yf + crop_size, xf - crop_size:xf + crop_size, :]
        crop_circle, likelihood = run_circle_max(crop_image, radius=he_radius, max_n=1, step=2)
        crop_circles.append(crop_circle)
        likelihoods.append(likelihood)
    if circle_likelihood is None:
        circle_likelihood = np.median(np.asarray(likelihood))
    detected_circles=[]
    for id in f_indices:
        crop_circle = crop_circles[id]
        likelihood = likelihoods[id]
        if likelihood > circle_likelihood:
            crop_circle[0] = crop_circle[0] + transposed_fiducial[id, 0] - crop_size
            crop_circle[1] = crop_circle[1] + transposed_fiducial[id, 1] - crop_size
            detected_circles.append(crop_circle)
    return image, detected_circles


# ------------------------------------------
#                data loading
# ------------------------------------------

# imagepath = '/home/huifang/workspace/data/humanpilot/'
# imagepath = '/home/huifang/workspace/data/mouse/'
root_path = '/home/huifang/workspace/data/fiducial_train/'
dataset_names = os.listdir(root_path)
for dataset_name in dataset_names:
    # dataset_name = 'mouse'
    print('-------------'+dataset_name+'-------------')
    temp = dataset_name.split('_')
    if len(temp)>1:
        he_radius = int(temp[-1])
    else:
        he_radius = H_RADIUS
    image_names= os.listdir(root_path+dataset_name)
    for image_name in image_names:
        # image_name='spatial4'
        print(image_name+'...')
        ##### run single image #####
        image, detected_circles = run(root_path + dataset_name + '/', image_name, circle_likelihood=10)

        ##### run iteratively #####
        # try:
        #     image, detected_circles = run(root_path+dataset_name+'/', image_name,circle_likelihood=10)
        # except:
        #     print("auto detection failed")
        #     continue
        ##### Save to file #####
        # save_path = root_path+dataset_name+'/'+ image_name+'/masks/'
        # os.makedirs(save_path,exist_ok=True)
        #
        # mask = np.zeros(image.shape[:2])
        # save_mask_to_file(mask,detected_circles,save_path+'hough_mask_solid_r2',-1)
        # mask = np.zeros(image.shape[:2])
        # save_mask_to_file(mask, detected_circles, save_path + 'hough_mask', 1)
        # mask = np.zeros(image.shape[:2])
        # save_mask_to_file(mask, detected_circles, save_path + 'hough_mask_w2', 2)
        # print('done')
        #####Visualization#####
        for circle in detected_circles:
            cv2.circle(image, (circle[0], circle[1]), circle[2], 1, 1)
        plt.imshow(image)
        plt.show()


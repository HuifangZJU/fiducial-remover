from __future__ import division
import cv2
import numpy as np
import time
from numba import jit
from matplotlib import pyplot as plt
import utils.icp as icp
from PIL import Image, ImageFilter
from scipy import stats
from PIL import Image
import seaborn as sns
import numpy as np
from PIL import Image
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm


@jit
def fill_acc_array_with_weight(acc_array,edges,height,width,radius_range, weight):
    for i in range(0, len(edges[0])):
        x0 = edges[0][i]
        y0 = edges[1][i]
        w0 = weight[x0,y0]
        for i in range(len(radius_range)):
            x = radius_range[i]
            y = 0
            while (y < x):
                if (x + x0 < height and y + y0 < width):
                    acc_array[x + x0, y + y0, i] += w0;  # Octant 1
                if (y + x0 < height and x + y0 < width):
                    acc_array[y + x0, x + y0, i] += w0;  # Octant 2
                if (-x + x0 < height and y + y0 < width):
                    acc_array[-x + x0, y + y0, i] += w0;  # Octant 4
                if (-y + x0 < height and x + y0 < width):
                    acc_array[-y + x0, x + y0, i] += w0;  # Octant 3
                if (-x + x0 < height and -y + y0 < width):
                    acc_array[-x + x0, -y + y0, i] += w0;  # Octant 5
                if (-y + x0 < height and -x + y0 < width):
                    acc_array[-y + x0, -x + y0, i] += w0;  # Octant 6
                if (x + x0 < height and -y + y0 < width):
                    acc_array[x + x0, -y + y0, i] += w0;  # Octant 8
                if (y + x0 < height and -x + y0 < width):
                    acc_array[y + x0, -x + y0, i] += w0;  # Octant 7
                if np.power(x,2) + np.power(y+1,2) > np.power(radius_range[i],2):
                    x = x-1
                y = y+1

def get_circle_pixels(circle_center,radius):
    x0 = circle_center[0]
    y0 = circle_center[1]
    pixels=[]

    x = radius
    y = 0
    while (y < x):
        pixels.append([x + x0, y + y0])
        pixels.append([y + x0, x + y0])
        pixels.append([-x + x0, y + y0])
        pixels.append([-y + x0, x + y0])
        pixels.append([-x + x0, -y + y0])
        pixels.append([-y + x0, -x + y0])
        pixels.append([x + x0, -y + y0])
        pixels.append([y + x0, -x + y0])
        if np.power(x, 2) + np.power(y + 1, 2) > np.power(radius, 2):
            x = x - 1
        y = y + 1
    pixels = np.asarray(pixels)
    return [pixels[:,0],pixels[:,1]]

def get_vector_distances(src,dst):
    dst_array = np.repeat(dst[:, np.newaxis], src.shape[0], axis=1)
    src_array = np.repeat(src[:, np.newaxis], dst.shape[0], axis=1)
    src_array = np.transpose(src_array, [1, 0])

    distance = np.power((dst_array - src_array), 2)
    distance = np.sqrt(distance)

    return distance
def get_matrix_distances(src,dst):
    dst_array = np.repeat(dst[:, :, np.newaxis], src.shape[0], axis=2)
    src_array = np.repeat(src[:, :, np.newaxis], dst.shape[0], axis=2)
    src_array = np.transpose(src_array, [2, 1, 0])

    distance = np.power((dst_array - src_array), 2)
    distance = np.sqrt(distance)
    return distance

def get_hough_lines(image):

    # Classic straight-line Hough transform
    h, theta, d = hough_line(image)
    peaks = hough_line_peaks(h, theta, d)

    # Generating figure 1
    LOCAL_DEBUG=True
    if LOCAL_DEBUG:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(np.log(1 + h),
                     extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                     cmap=cm.gray, aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image, cmap=cm.gray)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        # for _, angle, dist in zip(h, theta, d):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        plt.tight_layout()
        plt.show()





def get_square_paras(img_height,img_width,points,shrink=5):
    img_width = int(np.ceil(img_width/shrink))
    img_height = int(np.ceil(img_height/shrink))
    points = np.floor(points / shrink)
    points = points.astype(int)
    capture_area_threshold = 1500/shrink



    circle_image = np.zeros([img_height,img_width])
    # for i in range(-1,2):
    #     for j in range(-1,2):
    #             circle_image[points[:, 1]+i, points[:, 0]+j] = 255
    circle_image[points[:, 1] , points[:, 0]] = 255
    # lines = get_hough_lines(circle_image)

    # Classic straight-line Hough transform
    circle_image = circle_image.astype(np.uint8)
    lines = cv2.HoughLines(circle_image, 1, np.pi / 180, 10)
    centery=[]
    centerx=[]
    for line in lines:
        rho, theta = line[0]
        # visualization
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        if theta < np.pi/2:
            x1 = int(x0 + img_height * (-b))
            y1 = int(y0 + img_width * (a))
            x2 = int(x0 - 0*img_height * (-b))
            y2 = int(y0 - 0*img_width * (a))
        else:
            x1 = int(x0 + 0 * (-b))
            y1 = int(y0 + 0 * (a))
            x2 = int(x0 - img_height * (-b))
            y2 = int(y0 - img_width * (a))
        centerx.append((x1+x2)/2)
        centery.append((y1+y2)/2)
        # cv2.line(circle_image, (x1, y1), (x2, y2), 255, 1)
    print(centery)
    print(centerx)
    # plt.imshow(circle_image)
    # plt.show()

    horizon_lines = []
    vertical_lines = []
    angle_threshold=0.5
    for line,cx,cy in zip(lines,centerx,centery):
        rho, theta = line[0]
        print(theta)
        if np.abs(theta) < angle_threshold or np.abs(theta-np.pi) < angle_threshold:
            similar_flag = False
            for line in horizon_lines:
                if np.abs(line - cx) <2:
                    similar_flag = True
            if not similar_flag:
                horizon_lines.append(cx)
        elif np.abs(theta - np.pi / 2) < angle_threshold:
            similar_flag = False
            for line in vertical_lines:
                if np.abs(line - cy) < 2:
                    similar_flag = True
            if not similar_flag:
                vertical_lines.append(cy)

        else:
            continue

    horizon_lines = np.sort(np.abs(np.asarray(horizon_lines)))
    vertical_lines = np.sort(np.abs(np.asarray(vertical_lines)))

    hor_dis = np.triu(get_vector_distances(horizon_lines, horizon_lines))
    ver_dis= np.triu(get_vector_distances(vertical_lines, vertical_lines))

    hor_dis_vector_id = np.where(hor_dis>0)
    hor_dis_vector = hor_dis[hor_dis_vector_id]
    ver_dis_vector_id = np.where(ver_dis > 0)
    ver_dis_vector = ver_dis[ver_dis_vector_id]

    dis_dis = get_vector_distances(hor_dis_vector, ver_dis_vector)
    # plt.show()
    # square shape, horizon_dis ~= vertical_dis
    equal_id = np.where(dis_dis < 10)
    # capture area larger than 1000*1000
    greater_id = np.where((ver_dis_vector[equal_id[0]] > capture_area_threshold)
                          & (hor_dis_vector[equal_id[1]]>capture_area_threshold))

    final_verdis_id = equal_id[0][greater_id]
    final_hordis_id = equal_id[1][greater_id]

    ver_id_side0 = ver_dis_vector_id[0][final_verdis_id]
    ver_id_side1 = ver_dis_vector_id[1][final_verdis_id]
    hor_id_side0 = hor_dis_vector_id[0][final_hordis_id]
    hor_id_side1 = hor_dis_vector_id[1][final_hordis_id]

    #cases when a same pair of line have similar distance to different pairs of line
    ver_lines_upper = np.unique(vertical_lines[ver_id_side0])
    ver_lines_lower = np.unique(vertical_lines[ver_id_side1])
    hor_lines_upper = np.unique(horizon_lines[hor_id_side0])
    hor_lines_lower = np.unique(horizon_lines[hor_id_side1])

    ver_lines_upper = ver_lines_upper.astype(int)
    ver_lines_lower = ver_lines_lower.astype(int)
    hor_lines_upper = hor_lines_upper.astype(int)
    hor_lines_lower = hor_lines_lower.astype(int)
    print(ver_lines_lower)
    print(ver_lines_upper)
    print(hor_lines_upper)
    print(hor_lines_lower)

    scale = []
    framecenter_x = []
    framecenter_y = []
    if len(ver_lines_upper)>0:
        ver_min = np.min(ver_lines_upper)
        if len(ver_lines_lower) > 0:
            ver_max = np.max(ver_lines_lower)
            scale.append(ver_max - ver_min)
            framecenter_x = int((ver_max + ver_min) / 2)
    if len(hor_lines_upper)>0:
        hor_min = np.min(hor_lines_upper)
        if len(hor_lines_lower)>0:
            hor_max = np.max(hor_lines_lower)
            scale.append(hor_max - hor_min)
            framecenter_y = int((hor_max + hor_min) / 2)

    if scale:
        square_scale = np.mean(np.asarray(scale))
    else:
        square_scale = []
    LOCAL_DEBUG = True
    if LOCAL_DEBUG:
        for vmax in ver_lines_upper:
            for vmin in ver_lines_lower:
                for hmax in hor_lines_upper:
                    for hmin in hor_lines_lower:
                        cv2.line(circle_image, ( hmin,vmin), (hmin,vmax), 255, 1)
                        cv2.line(circle_image, (hmax,vmin), (hmax,vmax), 255, 1)
                        cv2.line(circle_image, (hmin,vmin), (hmax,vmin), 255, 1)
                        cv2.line(circle_image, (hmin,vmax), (hmax,vmax), 255, 1)
        if framecenter_y and framecenter_x:
            cv2.circle(circle_image, (framecenter_y,framecenter_x), radius=2, color=(255, 0, 0), thickness=-1)
        plt.imshow(circle_image)
        plt.show()

    return square_scale,centerx,centery

# @jit
def maxpooling_in_position(image,position,kernel_size):
    height, width = image.shape
    for i in range(position[0].shape[0]):
        x = position[0][i]
        y = position[1][i]
        lower_bound_x = x-kernel_size if x>kernel_size else 0
        lower_bound_y = y-kernel_size if y>kernel_size else 0

        bound_x = x+kernel_size if x<height-kernel_size else height
        bound_y = y+kernel_size if y<width-kernel_size else width

        temp = image[lower_bound_x:bound_x,lower_bound_y:bound_y]

        circle_temp = np.where(temp == temp.max())
        if len(circle_temp[0]) > 1:
            circle_center = (np.array([circle_temp[0][0]]), np.array([circle_temp[1][0]]))
        else:
            circle_center = circle_temp

        temp2 = np.zeros(temp.shape)
        temp2[circle_center] = temp.max()
        image[lower_bound_x:bound_x, lower_bound_y:bound_y] = temp2

def normalize_array(input):
    min_value = input.min()
    max_value = input.max()
    return (input-min_value)/(max_value-min_value)


# Using OpenCV Canny Edge detector to detect edges
def getBluredImg(original_image):
    blur_image = cv2.GaussianBlur(original_image, (3, 3), 0)
    blur_image = 255 * blur_image
    blur_image = np.uint8(blur_image)
    return blur_image

def getEdgedImg(blur_image,style):
    if style == "canny":
        # tiff: (250, 380), hires:(180,50)
        edged_image = cv2.Canny(blur_image, 180, 50)
    else:
        assert(style == "Image")
        blur_image = Image.fromarray(blur_image.astype(np.uint8))
        blur_image = blur_image.convert("L")
        edged_image= blur_image.filter(ImageFilter.FIND_EDGES)
    return np.asarray(edged_image)

def run_circle_threhold(original_image,radius,circle_threshold,step=2):
    # Gaussian Blurring of Gray Image
    original_image = normalize_array(original_image)
    blur_image=getBluredImg(original_image)
    edged_image = getEdgedImg(blur_image,"canny")


    edges = np.where(edged_image == 255)
    height, width = edged_image.shape
    radius_range = np.arange(radius-step,radius+step)

    edged_image = normalize_array(edged_image)
    acc_array = np.zeros((height, width, len(radius_range)))
    fill_acc_array_with_weight(acc_array, edges, height, width, radius_range,edged_image)

    candidate_centers = acc_array.max(axis=2)
    candidate_radius= acc_array.argmax(axis=2)


    circle_center = np.where(candidate_centers>circle_threshold)

    # remove near centers with maxpooling
    maxpooling_in_position(candidate_centers,circle_center,fiducial_radius)
    circle_center = np.where(candidate_centers > circle_threshold)

    # find corresponding radius for detected circles
    radius_index = candidate_radius[circle_center]
    radius_index = np.asarray(radius_index)
    radius = radius_index.copy()
    for i in range(len(radius_range)):
        radius = np.where(radius_index==i,radius_range[i],radius)
    circle_center = np.array(circle_center).transpose()
    circle_center = np.flip(circle_center,1)
    radius = radius[:,np.newaxis]
    circles = np.concatenate((circle_center,radius),axis=1)

    return circles


def run_circle_max(image,crop_image,radius,max_n,step=1):
    crop_image = normalize_array(crop_image)
    blur_image = getBluredImg(crop_image)

    LOCAL_DEBUG=False

    blur_image= blur_image * 255
    edged_image = getEdgedImg(blur_image,"Image")

    edged_image = normalize_array(edged_image)
    edged_image[0,:]=0
    edged_image[-1, :] = 0
    edged_image[:, 0] = 0
    edged_image[:, -1] = 0
    edges = np.where(edged_image > 0)

    # edged_image = crop_image

    radius_range = np.arange(radius - step, radius+1)
    height, width = edged_image.shape
    acc_array = np.zeros(((height, width, len(radius_range))))
    fill_acc_array_with_weight(acc_array, edges, height, width, radius_range,edged_image)


    candidate_radius_index = acc_array.argmax(axis=2)
    candidate_centers = acc_array.max(axis=2)
    # candidate_centers = np.where(candidate_centers == candidate_centers.max(),candidate_centers,0)

    ##fiduicial circle shape mask
    # column=np.arange(0,candidate_centers.shape[0])
    # columns = np.repeat(column[:,np.newaxis],candidate_centers.shape[1],axis=1)
    # rows = columns.transpose()
    # columns = columns-fiducial_radius
    # rows = rows-fiducial_radius
    # distance = np.power(columns, 2) + np.power(rows, 2)
    # distance = np.sqrt(distance)
    # candidate_centers = np.where(distance<fiducial_radius+1,candidate_centers,0)


    #find top n elements in candidate_centers
    # candidate_centers_vector = np.asarray(candidate_centers).copy()
    # candidate_centers_vector = candidate_centers_vector.reshape(-1)
    # ind = np.argpartition(candidate_centers_vector, -max_n)[-max_n:]
    # max_values = candidate_centers_vector[ind]

    circle_center = np.where(candidate_centers == candidate_centers.max())
    if len(circle_center[0])>1:
        circle_center = (np.array([circle_center[0][0]]),np.array([circle_center[1][0]]))


    # find corresponding radius for detected circles
    radius_index = candidate_radius_index[circle_center]
    radius_index = np.asarray(radius_index)
    radius = radius_index.copy()
    for i in range(len(radius_range)):
        radius = np.where(radius_index == i, radius_range[i], radius)
    circle_center = np.array(circle_center).transpose()
    circle_center = np.flip(circle_center, 1)
    radius = radius[:, np.newaxis]
    circles = np.concatenate((circle_center, radius), axis=1)

    if LOCAL_DEBUG:
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(image)
        axarr[0, 1].imshow(crop_image)
        axarr[1, 0].imshow(candidate_centers)
        cv2.circle(image, (circles[0, 0], circles[0, 1]),circles[0,2], (0, 0, 250), 1)
        axarr[1, 1].imshow(image)
        print(circles[0, 0],circles[0, 1],circles[0,2])
        plt.show()
    return circles


def get_transposed_fiducials(circles,circles_f,shrink_scale, iter=1):
    transform=[]
    mean_error = 999
    circles_f[:,:2] = circles_f[:,:2]*shrink_scale
    for i in range(iter):
        num_circle_in_fiducials = circles_f.shape[0]
        num_circle_in_tissue = circles.shape[0]
        if num_circle_in_tissue > num_circle_in_fiducials:
            np.random.shuffle(circles)
            circle_center_select = circles[:num_circle_in_fiducials, :]
            circle_center_f_select = circles_f
        else:
            np.random.shuffle(circles_f)
            circle_center_f_select = circles_f[:num_circle_in_tissue, :]
            circle_center_select = circles
            # use icp find alignment
        temp_transform, temp_error = icp.get_icp_transformation(circle_center_select[:,:2], circle_center_f_select[:,:2])
        if temp_error<mean_error:
           transform = temp_transform
    transposed_circle = icp.apply_icp_transformation(circles_f, transform)
    return transposed_circle.astype(int)

def find_nearest_points(src,dst):
    dst_array = np.repeat(dst[:, :, np.newaxis], src.shape[0], axis=2)
    src_array = np.repeat(src[:, :, np.newaxis], dst.shape[0], axis=2)
    src_array = np.transpose(src_array, [2, 1, 0])

    distance = np.power((dst_array - src_array), 2)
    distance = distance[:, 0, :] + distance[:, 1, :]
    distance = np.sqrt(distance)
    indices = np.argmin(distance, axis=1)
    return indices, distance

def save_image(array,filename,format="RGB"):
    if array.max()<1.1:
        array = 255 * array
    array = array.astype(np.uint8)
    array = Image.fromarray(array)
    if format == "RGB":
        array = array.convert('RGB')
    else:
        assert(format == "L")
        array = array.convert('L')
    array.save(filename)


def get_refined_centers(circle_center,transposed_fiducial,distance,indices,distance_threshold=10):
    LOCAL_DEBUG = False
    SAVE_FILE = False
    new_center=[]
    distance_aligned_center=[]
    fiducial_refined_center=[]
    distance_missed_fiducial=[]
    missed_circle=0
    if LOCAL_DEBUG:
        true_positive_distance=[]
        all_distance=[]
    if SAVE_FILE:
        fileid=0
        testid=34
    for i,idx in zip(np.arange(distance.shape[0]),indices):
        if LOCAL_DEBUG:
            all_distance.append(distance[i, idx])
        if distance[i,idx]<0:
        # if distance[i,idx]<distance_threshold:
            new_center.append(circle_center[idx,:])
            distance_aligned_center.append(circle_center[idx,:])
            if LOCAL_DEBUG:
                true_positive_distance.append(distance[i,idx])
        else:
            crop_size = 2*fiducial_radius
            x = transposed_fiducial[i, 0]
            y = transposed_fiducial[i, 1]
            r = transposed_fiducial[i, 2]
            distance_missed_fiducial.append([x,y,r])

            crop_image = image[y - crop_size:y + crop_size, x - crop_size:x + crop_size, :]
            # pix2pix_image = Image.open('/home/huifang/workspace/code/fiducial_remover/pix2pix/test/151507_pix2.png').convert('L')
            # pix2pix_image = np.asarray(pix2pix_image)
            # crop_image_pix2pix = pix2pix_image[y - crop_size:y + crop_size, x - crop_size:x + crop_size]
            # crop_tiff = get_local_tiff(image_tiff,x,y,crop_size)
            # f,a=plt.subplots(1,2)
            # a[0].imshow(crop_image)
            # a[1].imshow(crop_tiff)
            # plt.show()
            crop_circles = run_circle_max(crop_image,crop_image,radius=he_radius, max_n=1,step=2)
            if crop_circles.size == 0:
                missed_circle += 1
            else:
                if SAVE_FILE & (distance[i,idx]<distance_threshold):
                # if SAVE_FILE:
                    crop_image_show = crop_image.copy()
                    mask = np.ones(crop_image.shape[0:2])
                    # crop_image_show = crop_image_show*255
                    cv2.circle(crop_image_show, (crop_circles[0, 0], crop_circles[0, 1]), crop_circles[0, 2], (1,1,1), 1)
                    cv2.circle(mask, (crop_circles[0, 0], crop_circles[0, 1]), crop_circles[0, 2], 0, 1)
                    # f,axarr = plt.subplots(1,3)
                    # axarr[0].imshow(crop_image)
                    # axarr[1].imshow(crop_image_show)
                    # axarr[2].imshow(mask,cmap='gray')
                    # # plt.show()
                    # plt.draw()
                    # plt.waitforbuttonpress()
                    # plt.close()
                    # print("Save or not: ")
                    # label = input()
                    label="1"
                    if label == "1":
                        save_image(crop_image,'../../data/crop_images/train/151507/%s_image.png' % fileid, format="RGB")
                        save_image(mask, '../../data/crop_images/train/151507/%s_mask.png' % fileid, format="L")
                        # save_image(crop_image_show, '../../data/crop_images/reconstruction/%s_mask.jpeg' % fileid, format="RGB")
                        fileid +=1
                        print("done")
                    else:
                        if label == "2":
                            save_image(crop_image, '../../data/crop_images/test/%s_image.png' % testid, format="RGB")
                            testid +=1
                            print("done")

                new_center.append([crop_circles[0,0]+x-crop_size,crop_circles[0,1]+y-crop_size,crop_circles[0,2]])
                fiducial_refined_center.append([crop_circles[0,0]+x-crop_size,crop_circles[0,1]+y-crop_size,crop_circles[0,2]])
    if LOCAL_DEBUG:
        true_positive_distance = np.asarray(true_positive_distance)
        true_positive_distance.sort()
        all_distance = np.asarray(all_distance)
        all_distance.sort()
        sns.displot(true_positive_distance,bins=30)
        plt.show()

    print("missed %s circles" % missed_circle)

    return np.asarray(new_center), np.asarray(distance_aligned_center), \
           np.asarray(fiducial_refined_center),np.asarray(distance_missed_fiducial)



def get_local_tiff(image_tiff,x,y,crop_size):
    hires_to_tiff = 0.150015
    x_tiff = int(x / hires_to_tiff)
    y_tiff = int(y / hires_to_tiff)
    crop_size_tiff = int(crop_size/hires_to_tiff)

    image_tiff_crop= image_tiff[y_tiff - crop_size_tiff:y_tiff + crop_size_tiff, x_tiff - crop_size_tiff:x_tiff + crop_size_tiff, :]
    return image_tiff_crop


def save_mask_to_file(image,center,filename):
    for i in range(center.shape[0]):
        # cv2.circle(image, (center[i, 0], center[i, 1]), center[i, 2]+1, 0, -1)
        cv2.circle(image, (center[i, 0], center[i, 1]), fiducial_radius, 0, -1)
    save_image(image, filename, format="L")

    with open(filename+'.txt', 'w') as txt_file:
        for line in center:
            txt_file.write(" ".join([str(n) for n in line]) + "\n")

# ------------------------------------------
#                data loading
# ------------------------------------------
#Read data
#todo
# 151673 75
# image = plt.imread('../../data/humanpilot/151673/spatial/tissue_hires_image.png')
# image = plt.imread('/home/huifang/workspace/code/fiducial_remover/pix2pix/test/151507_pix2.png')

# image = plt.imread('../../data/mouse/mouse_hires/posterior_v1.png')
#failed cases: 1 3 6 8
#anotation cases: 2 7
image = plt.imread('/home/huifang/workspace/data/fiducial_he/spatial4/tissue_hires_image.png')
# outputfile = "./alignment/anterior_v2.png"
fiducials = plt.imread('./data/fiducials.jpeg')


DEBUG = False
# ------------------------------------------
#                circle detection
# ------------------------------------------
he_radius = 9
fiducial_radius = 15
circles = run_circle_threhold(image,radius=he_radius,circle_threshold=30)
circles_f = run_circle_threhold(fiducials,radius=fiducial_radius,circle_threshold=50)
if not DEBUG:
    output = image.copy()
    for i in range(circles.shape[0]):
        cv2.circle(output, (circles[i,0], circles[i,1]), circles[i,2], (0, 255, 0), 2)
    f, axarr = plt.subplots(2,3)
    plt.setp(axarr, xticks=[], yticks=[])
    axarr[0,0].imshow(image)
    axarr[0,1].imshow(output)
    axarr[0,2].imshow(image)
    axarr[0,2].scatter(circles[:,0],circles[:,1],marker='.',color="red",s=1)

    output_f = fiducials.copy()
    for i in range(circles_f.shape[0]):
        cv2.circle(output_f, (circles_f[i, 0], circles_f[i, 1]), circles_f[i, 2], (0, 255, 0), 2)
    axarr[1,0].imshow(fiducials)
    axarr[1,1].imshow(output_f)
    axarr[1,2].imshow(fiducials)
    axarr[1,2].scatter(circles_f[:,0],circles_f[:,1],marker='.',color="red",s=1)
    plt.show()

circle_scale,tissue_center_x,tissue_center_y = get_square_paras(image.shape[0],image.shape[1],circles)
# fiducial_scale,fiducial_center_x,fiducial_center_y = get_square_paras(fiducials.shape[0],fiducials.shape[1],circles_f)
# SHRINK = circle_scale/fiducial_scale
#
#
# # -------------------------------------------------------
# #          find alignment to fiducial based on ICP
# # -------------------------------------------------------
#
# transposed_fiducial = get_transposed_fiducials(circles,circles_f,shrink_scale =SHRINK,iter=50)
#
# # mask = np.ones(image.shape[0:2])
# # save_mask_to_file(mask,transposed_fiducial,'151507_mask_template.png')
#
# indices, distance = find_nearest_points(circles[:,:2],transposed_fiducial[:,:2])
#
#
# if DEBUG:
#     f2, axarr2 = plt.subplots(1,3)
#     plt.setp(axarr2, xticks=[], yticks=[])
#     axarr2[0].scatter(circles[:, 0], circles[:, 1])
#     axarr2[0].axis('equal')
#     axarr2[1].scatter(circles[:, 0], circles[:, 1])
#     axarr2[1].scatter(transposed_fiducial[:, 0], transposed_fiducial[:, 1])
#     axarr2[1].axis('equal')
#     output_fiducial = image.copy()
#     for i in range(transposed_fiducial.shape[0]):
#         cv2.circle(output_fiducial, (transposed_fiducial[i, 0], transposed_fiducial[i, 1]), fiducial_radius, (0, 255, 0), 2)
#     axarr2[2].imshow(output)
#     plt.show()
#
# # -------------------------------------------------------
# #         find new circle centers based on alignment
# # -------------------------------------------------------
# new_center, aligned_center,refined_center, missed_fiducial = \
#     get_refined_centers(circles,transposed_fiducial,distance,indices,distance_threshold=he_radius)
#
#
#
# if not DEBUG:
#     aligned_output = image.copy()
#     aligned_output_refine = image.copy()
#     for i in range(aligned_center.shape[0]):
#         cv2.circle(aligned_output, (aligned_center[i, 0], aligned_center[i, 1]), aligned_center[i, 2], (0, 250, 0),2)
#     for i in range( missed_fiducial.shape[0]):
#         cv2.circle(aligned_output, (missed_fiducial[i, 0],  missed_fiducial[i, 1]),  missed_fiducial[i, 2],(255, 0, 0), 2)
#     for i in range(refined_center.shape[0]):
#         cv2.circle(aligned_output, (refined_center[i, 0], refined_center[i, 1]), refined_center[i, 2], (0, 0, 255), 2)
#     for i in range(new_center.shape[0]):
#         cv2.circle(aligned_output_refine, (new_center[i,0], new_center[i,1]), new_center[i,2], (0,250,0), 2)
#     # f = plt.figure(figsize=(30,30))
#     # plt.imshow(aligned_output_refine)
#     # plt.savefig(output)
#     # print("done")
#     f3, axarr3 = plt.subplots(1,2)
#     plt.setp(axarr3, xticks=[], yticks=[])
#     axarr3[0].imshow(aligned_output)
#     axarr3[1].imshow(aligned_output_refine)
#     plt.show()


# mask = np.ones(image.shape)
# for i in range(new_center.shape[0]):
#     cv2.circle(mask, (new_center[i, 0], new_center[i, 1]), new_center[i, 2], (0, 0, 0), 3)
# # mask = Image.fromarray(mask.astype(np.uint8))
# # mask = mask.convert('RGB')
# # mask.save('151509_mask_full.jpeg')
# # print('done')
#
# circle_pixels = np.where(mask[:,:,0] == 0)
# circle_pixels_x = circle_pixels[0]
# circle_pixels_y = circle_pixels[1]
#
# img_final = image.copy()
# for i in range(5):
#     for x,y in zip(circle_pixels_x,circle_pixels_y):
#         crop_size = he_radius
#         crop_image = image[x-crop_size:x+crop_size,y-crop_size:y+crop_size,:]
#         mean_r = np.mean(crop_image[:,:,0])
#         mean_g = np.mean(crop_image[:, :, 1])
#         mean_b = np.mean(crop_image[:, :, 2])
#         img_final[x,y,:] = [mean_r,mean_g,mean_b]
#
# if DEBUG:
#     f4,axarr4 = plt.subplots(1,2)
#     axarr4[0].imshow(mask)
#     axarr4[1].imshow(img_final)
# plt.show()








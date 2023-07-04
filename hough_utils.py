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
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data
from skimage import morphology
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
    # peaks = hough_line_peaks(h, theta, d)

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
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        plt.tight_layout()
        plt.show()

def get_square_lines(img_height,img_width,points):
    circle_image = np.zeros([img_height,img_width])
    circle_image[points[:, 0], points[:, 1]] = 255

    # Classic straight-line Hough transform
    circle_image = circle_image.astype(np.uint8)
    scale = int(np.floor(points.shape[0]/1000))
    if scale<1:
        scale=1

    lines=None
    try:
        while not lines:
            lines = cv2.HoughLines(circle_image, 1, np.pi / 180, 5*scale)
            scale = scale-1
    except:
        pass
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

    horizon_lines = []
    vertical_lines = []
    for line,cx,cy in zip(lines,centerx,centery):
        rho, theta = line[0]
        if np.abs(theta) < 0.1 or np.abs(theta-np.pi) < 0.1:
            similar_flag = False
            for line in horizon_lines:
                if np.abs(line - cx) <10:
                    similar_flag = True
            if not similar_flag:
                horizon_lines.append(cx)
        elif np.abs(theta - np.pi / 2) < 0.1:
            similar_flag = False
            for line in vertical_lines:
                if np.abs(line - cx) < 10:
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
    equal_id = np.where(dis_dis < 20)
    # capture area larger than 1000*1000
    greater_id = np.where((ver_dis_vector[equal_id[0]] > 1000) & (hor_dis_vector[equal_id[1]]>1000))

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

    return ver_lines_upper.astype(int),ver_lines_lower.astype(int),hor_lines_upper.astype(int),hor_lines_lower.astype(int)

def get_square_paras(img_height,img_width,circles,image):
    ver_lines_upper,ver_lines_lower,hor_lines_upper,hor_lines_lower = get_square_lines(img_height,img_width, circles)
    scale=[]
    framecenter_x=[]
    framecenter_y=[]
    if len(ver_lines_upper)>0 and len(ver_lines_lower)>0:
        vmin = np.min(ver_lines_upper)
        vmax = np.max(ver_lines_lower)
        scale.append(vmax-vmin)
        framecenter_x = int((vmax + vmin) / 2)
    if len(hor_lines_lower) and len(hor_lines_upper):
        hmin = np.min(hor_lines_upper)
        hmax = np.max(hor_lines_lower)
        scale.append(hmax - hmin)
        framecenter_y = int((hmax + hmin) / 2)
    if scale:
        # square_scale = np.mean(np.asarray(scale))
        LOCAL_DEBUG = True
        if LOCAL_DEBUG:
            output = image.copy()
            cv2.line(output, (vmin, hmin), (vmax, hmin), (255, 0, 0), 2)
            cv2.line(output, (vmin, hmax), (vmax, hmax), (255, 0, 0), 2)
            cv2.line(output, (vmin, hmin), (vmin, hmax), (255, 0, 0), 2)
            cv2.line(output, (vmax, hmin), (vmax, hmax), (255, 0, 0), 2)
            for vmax in ver_lines_upper:
                for vmin in ver_lines_lower:
                    for hmax in hor_lines_upper:
                        for hmin in hor_lines_lower:
                            cv2.line(output, (vmin, hmin), (vmax, hmin), (255, 0, 0), 2)
                            cv2.line(output, (vmin, hmax), (vmax, hmax), (255, 0, 0), 2)
                            cv2.line(output, (vmin, hmin), (vmin, hmax), (255, 0, 0), 2)
                            cv2.line(output, (vmax, hmin), (vmax, hmax), (255, 0, 0), 2)
            cv2.circle(output, (framecenter_x, framecenter_y), radius=10, color=(255, 0, 0), thickness=-1)
            plt.imshow(output)
            plt.show()

    return framecenter_x,framecenter_y,scale

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


def get_edge_pixels(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(thresh, 1000, 1000)

    kernel_d = np.ones((3, 3), dtype=np.uint8)
    kernel_d[0,0]=0
    kernel_d[0,2]=0
    kernel_d[2,0]=0
    kernel_d[2,2]=0

    edges_temp = cv2.erode(edges,kernel_d,iterations=1)
    edges = edges - edges_temp


    edges = cv2.dilate(edges, kernel_d, iterations=2)


    # edges = label(edges, neighbors=8)
    re = morphology.remove_small_objects(edges, min_size=30000)
    re = np.where(re > 0, 255, 0)
    re = re.astype(np.uint8)

    edges = cv2.erode(re, kernel_d, iterations=2)
    edges_temp = cv2.erode(edges, kernel_d, iterations=1)
    re = edges - edges_temp

    return re


def run_circle_threhold(original_image,_radius_,circle_threshold,edgemethod="canny",step=3):
    if edgemethod == 'self':
        edged_image = get_edge_pixels(original_image)
        plt.imshow(edged_image)
        plt.show()
    # # Gaussian Blurring of Gray Image
    else:
        original_image = normalize_array(original_image)
        blur_image=getBluredImg(original_image)
        edged_image = getEdgedImg(blur_image,"canny")
        plt.imshow(edged_image)
        plt.show()


    edges = np.where(edged_image == 255)
    height, width = edged_image.shape
    radius_range = np.arange(_radius_-step,_radius_+step)

    edged_image = normalize_array(edged_image)
    acc_array = np.zeros((height, width, len(radius_range)))
    fill_acc_array_with_weight(acc_array, edges, height, width, radius_range,edged_image)

    candidate_centers = acc_array.max(axis=2)
    candidate_radius= acc_array.argmax(axis=2)
    circle_center = np.where(candidate_centers>circle_threshold)

    # remove near centers with maxpooling
    maxpooling_in_position(candidate_centers,circle_center,_radius_)
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

def generate_mask(image_size,circles,circle_width):
    mask = np.zeros(image_size)
    for i in range(circles.shape[0]):
        cv2.circle(mask, (circles[i, 0], circles[i, 1]), circles[i, 2], 1, circle_width)
    return mask

def run_circle_max(crop_image,radius,max_n,step=1):
    crop_image = normalize_array(crop_image)
    blur_image = getBluredImg(crop_image)
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
    max_value = candidate_centers.max()
    circle_center = np.where(candidate_centers == max_value)
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


    LOCAL_DEBUG = False
    # LOCAL_DEBUG = True
    if LOCAL_DEBUG:
        image_show = crop_image.copy()
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(image_show)
        axarr[0, 1].imshow(crop_image)
        axarr[1, 0].imshow(candidate_centers)
        cv2.circle(image_show, (circles[0, 0], circles[0, 1]),circles[0,2], (250, 0, 0), 1)
        axarr[1, 1].imshow(image_show)
        print(circles[0, 0],circles[0, 1],circles[0,2])
        print(max_value)
        plt.show()
    return circles[0], max_value


def get_transposed_fiducials(circles,circles_f,iter=1):
    transform=[]
    mean_error = 999
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

def find_nearest_points(src,dst,is_same=False):
    dst_array = np.repeat(dst[:, :, np.newaxis], src.shape[0], axis=2)
    src_array = np.repeat(src[:, :, np.newaxis], dst.shape[0], axis=2)
    src_array = np.transpose(src_array, [2, 1, 0])

    distance = np.power((dst_array - src_array), 2)
    distance = distance[:, 0, :] + distance[:, 1, :]
    distance = np.sqrt(distance)
    if is_same:
        maxvalue = np.max(distance)
        for i in range(distance.shape[0]):
            distance[i,i]=maxvalue
    indices = np.argmin(distance, axis=1)
    distance = np.min(distance,axis=1)
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

def get_local_tiff(image_tiff,x,y,crop_size):
    hires_to_tiff = 0.150015
    x_tiff = int(x / hires_to_tiff)
    y_tiff = int(y / hires_to_tiff)
    crop_size_tiff = int(crop_size/hires_to_tiff)

    image_tiff_crop= image_tiff[y_tiff - crop_size_tiff:y_tiff + crop_size_tiff, x_tiff - crop_size_tiff:x_tiff + crop_size_tiff, :]
    return image_tiff_crop

def save_mask_to_file(image,circles,filename,width=1,format='npy'):
    if np.max(image) == 0.0:
        try:
            for i in range(circles.shape[0]):
                # cv2.circle(image, (center[i, 0], center[i, 1]), center[i, 2], 0, -1)
                cv2.circle(image, (circles[i, 0], circles[i, 1]), circles[i, 2], 1, width)
        except:
            for circle in circles:
                cv2.circle(image, (circle[0], circle[1]), circle[2], 1, width)
    save_image(image, filename + '.png', format="L")
    if format=='npy':
        np.save(filename + '.npy', np.array(circles))
    else:
        with open(filename+'.txt', 'w') as txt_file:
            for line in circles:
                txt_file.write(" ".join([str(n) for n in line]) + "\n")







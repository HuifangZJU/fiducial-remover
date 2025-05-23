import os
import json
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.signal import convolve2d
import fiducial_utils
from hough_utils import *
from fiducial_utils import *
from skimage.draw import disk
import math
import matplotlib.patches as patches
import random
import statistics
import matplotlib.path as mpltPath
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

def plot_color_circles_in_image(image,in_tissue_circles,out_tissue_circles, hard_circles,width):

    for circle in in_tissue_circles:
        cv2.circle(image, (circle[0],circle[1]), circle[2],[255,0,0], width)

    for circle in out_tissue_circles:

        cv2.circle(image, (circle[0],circle[1]), circle[2], [0, 255, 0], width)
    for circle in hard_circles:
        cv2.circle(image, (circle[0], circle[1]), circle[2], [0, 0, 255], width)
    return image

def plot_circles_in_image(image,circles,width):

    for circle in circles:
        cv2.circle(image, (circle[0],circle[1]), circle[2],[255,0,0], width)

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
                circle_radius = circle_radius+1
                circle_rect = (circle_y - circle_radius,circle_x - circle_radius,  2 * circle_radius, 2 * circle_radius)


                if rectangles_intersect(patch_rect, circle_rect):
                    annotation[i, j] = 1
                    # plt.imshow(image[patch_x:patch_x + step, patch_y:patch_y + step])
                    # print(annotation[i, j])
                    # plt.show()
                    break
    return annotation

def get_shape_from_annotation_path(fileid):
    # Replace with the path to your JSON file
    labelme_json_path =  './location_annotation/'+str(fileid) +'.json'

    # Read the JSON file
    with open(labelme_json_path, 'r') as file:
        labelme_data = json.load(file)

    # Initialize variables to store the 'outer' and 'inner' polygons
    outer_polygon = None
    inner_polygon = None

    # Parse the JSON data for 'outer' and 'inner' polygon information
    for shape in labelme_data['shapes']:
        if shape['label'].lower() == 'outer' and shape['shape_type'] == 'polygon':
            # Assuming polygons are represented by 4 points (x, y)
            outer_polygon = np.array(shape['points'], dtype=np.int32)
        elif shape['label'].lower() == 'inner' and shape['shape_type'] == 'polygon':
            # Assuming polygons are represented by 4 points (x, y)
            inner_polygon = np.array(shape['points'], dtype=np.int32)
    # Now `bounding_boxes` contains all the bounding box information
    return outer_polygon,inner_polygon


def calculate_distance2(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def read_hard_circles_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    circles = []
    for shape in data['shapes']:
        if shape['label'] == 'circle':
            center = np.array(shape['points'][0])
            circumference_point = np.array(shape['points'][1])

            # Calculate the radius
            radius = calculate_distance2(center, circumference_point)
            center_with_radius = np.append(center, radius)

            circles.append(np.append(center_with_radius, 2))
    circles = np.asarray(circles).astype(int)
    return circles

def get_position_mask(img_size,i):
    outer_polygon, inner_polygon = get_shape_from_annotation_path(i)
    mask = np.zeros(img_size, dtype=np.uint8)
    cv2.fillPoly(mask, [outer_polygon], color=1)
    # Fill the inner polygon with zeros (black) to create the ring effect
    cv2.fillPoly(mask, [inner_polygon], color=0)
    mask = mask.astype(float)
    return mask

def divide_mask_into_patches(mask, patch_size):

    height, width = mask.shape

    # Calculate the number of patches in both dimensions
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    # Initialize an array to store the patch labels
    patch_labels = np.zeros((num_patches_y, num_patches_x), dtype=np.uint8)
    patch_fullsize_labels = np.zeros((height, width), dtype=np.uint8)

    # Iterate over the mask and label each patch
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            patch = mask[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size]
            label = 1 if np.any(patch == 1) else 0
            patch_labels[y, x] = label
            if label==1:
                patch_fullsize_labels[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size] = np.ones((patch_size, patch_size), dtype=np.uint8)
            else:
                patch_fullsize_labels[y * patch_size:(y + 1) * patch_size,
                x * patch_size:(x + 1) * patch_size] = np.zeros((patch_size, patch_size), dtype=np.uint8)
    return patch_labels,patch_fullsize_labels

def get_circles_from_annotation_path(image_name):
    annotation_path = get_annotation_path(image_name)
    print(annotation_path)

    # if not os.path.exists(annotation_path):
    #     continue
    in_tissue_circles, out_tissue_circles = get_annotated_circles(annotation_path)
    if len(unique_pairs_below_threshold(in_tissue_circles, 10)) > 0:
        in_tissue_circles = remove_overlapping_circles(in_tissue_circles, 10)
    if len(unique_pairs_below_threshold(out_tissue_circles, 10)) > 0:
        out_tissue_circles = remove_overlapping_circles(out_tissue_circles, 10)
    circles = in_tissue_circles + out_tissue_circles

    return in_tissue_circles,out_tissue_circles,np.asarray(circles)

def get_circles_from_file(image_name):
    circles = np.load(image_name.split('.')[0]+'.npy')
    in_tissue_circles = [circle for circle in circles if circle[-1] == 1]
    out_tissue_circles = [circle for circle in circles if circle[-1] == 0]
    hard_circles = [circle for circle in circles if circle[-1] == 2]
    return in_tissue_circles,out_tissue_circles,hard_circles,circles

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

def random_select_percentage_elements(input_array, percent_to_select):
    num_to_select = int((percent_to_select / 100) * input_array.shape[0])
    selected_indices = np.random.choice(input_array.shape[0], num_to_select, replace=False)
    selected_elements = input_array[selected_indices]
    return selected_elements
def random_select_number_elements(input_array,num_to_select):
    selected_indices = np.random.choice(input_array.shape[0], num_to_select, replace=False)
    selected_elements = input_array[selected_indices]
    return selected_elements

def find_circles_in_polygon(circles, inner_polygon):
    """
    Find which circles are inside the given polygon.

    :param circles: ndarray of shape (n, 3), where each row is (x, y, radius)
    :param inner_polygon: ndarray of shape (4, 2), representing a polygon
    :return: ndarray of circles inside the polygon
    """
    polygon_path = mpltPath.Path(inner_polygon)
    centers = circles[:, :2]  # Extract the center coordinates of the circles
    is_inside = polygon_path.contains_points(centers)
    return circles[is_inside]

def find_circles_in_position(circles, inner_polygon,outter_polygon):
    """
    Find which circles are inside the given polygon.

    :param circles: ndarray of shape (n, 3), where each row is (x, y, radius)
    :param inner_polygon: ndarray of shape (4, 2), representing a polygon
    :return: ndarray of circles inside the polygon
    """
    centers = circles[:, :2]  # Extract the center coordinates of the circles
    inner_polygon_path = mpltPath.Path(inner_polygon)
    outer_polygon_path = mpltPath.Path(outter_polygon)
    is_inposition = outer_polygon_path.contains_points(centers) & ~inner_polygon_path.contains_points(centers)
    return circles[is_inposition]

def get_annotation_path(imagepath):
    dataset = imagepath.split('/')[6]
    index = imagepath.split('/')[7]
    index = index.split('.')
    index = index[0]
    data_path = SAVE_ROOT + dataset + '_' + index
    return data_path

def check_annotation(image_name):
    in_tissue_circles,out_tissue_circles,circles = get_circles_from_annotation_path(image_name)
    if os.path.exists('./circle_annotation/'+str(i)+'.json'):
        hard_circles = read_hard_circles_from_json('./circle_annotation/'+str(i)+'.json')
    else:
        return
    circles = np.vstack((circles, hard_circles))
    np.save(image_name.split('.')[0] + '.npy', circles)
    print('saved')


def plot_pie_chart(numbers):
    # Calculate annotation percentages by dividing each number by 555
    percentages = [(num / 555) * 100 for num in numbers]

    # Initialize counters for different percentage ranges
    percent_100 = 0
    percent_90_100 = 0
    percent_80_90 = 0
    percent_70_80 = 0
    percent_below_70 = 0

    # Categorize percentages into different ranges
    for percent in percentages:
        if percent == 100:
            percent_100 += 1
        elif 95 <= percent < 100:
            percent_90_100 += 1
        elif 85 <= percent < 95:
            percent_80_90 += 1
        elif 70 <= percent < 85:
            percent_70_80 += 1
        else:
            percent_below_70 += 1

    # Create data for the pie chart
    labels = ['No deficiency', '0-5%', '5%-15%', '15%-30%', 'More than 30%']
    sizes = [percent_100, percent_90_100, percent_80_90, percent_70_80, percent_below_70]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightpink']
    explode = (0.1, 0, 0, 0, 0)  # Explode the 1st slice (100%)

    # Create the pie chart
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
    # Draw a circle in the center of the pie chart (optional)
    # centre_circle = plt.Circle((0, 0), 0.30, fc='white')
    # fig = plt.gcf()
    # fig.gca().add_artist(centre_circle)

    # Set the title above the pie chart
    plt.title("Distribution of Annotation Deficiency", y=1.08)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Show the pie chart
    plt.show()



source_images = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
f = open(source_images, 'r')
fiducial_images = f.readlines()
SAVE_FILE = False
if SAVE_FILE:
    # imagelist_path = '/home/huifang/workspace/data/imagelists/st_upsample_trainable_images_final_with_location_frame.txt'
    dest_train_images = '/home/huifang/workspace/data/imagelists/st_cellpose_trainable_images.txt'
    dest_test_images = '/home/huifang/workspace/data/imagelists/st_cellpose_test_images.txt'
    f_list_train = open(dest_train_images,'w')
    f_list_test = open(dest_test_images,'w')

width = 2
patch_size = 32
auto_cnt=0
# auto_good_shape_id = [5,13,16,20,21,34,35,39,40,45,47,54,59,60,63,107,120]
cellpose_good_shape_id=[4, 5, 14, 15, 21, 34, 39, 56, 59, 60, 63, 66,93, 106,108,129,130]
# shape_regulated_bad_shape_id=[45,67,68,69,70,84,92,94,97,100,102,108,111,112,116,121,126,127,128,130,132,133,135,144,146,160,166]
for i in range(0,len(fiducial_images)):

    # if i in badfile:
    #     continue

    image_name = fiducial_images[i].split(' ')[0]
    groupid = fiducial_images[i].split(' ')[2]
    image_name = image_name.rstrip('\n')

    # f_temp.write(image_name)

    print(str(len(fiducial_images))+'---'+str(i))
    # if not os.path.exists(image_name.split('.')[0]+ '_auto.npy'):
    #     f_list_test.write(fiducial_images[i])
    #     continue
        # continue
    # auto = plt.imread(image_name.split('.')[0]+ '_auto.png')
    # shape_regulated = plt.imread(image_name.split('.')[0]+ '_shape_constrained.png')
    # f,a = plt.subplots(1,2)
    # a[0].imshow(auto)
    # a[1].imshow(shape_regulated)
    # plt.show()
    # continue
    image = plt.imread(image_name)

    # print(image_name)
    # annotation_image = get_position_mask(image.shape[:2], i)
    # patches,patch_full_size = divide_mask_into_patches(annotation_image, patch_size)
    #
    # save_image(annotation_image, image_name.split('.')[0]+ '_binary_gt.png', format="L")
    # save_image(patch_full_size, image_name.split('.')[0] + '_binary_patch.png', format="L")
    #
    # continue

    circles = np.load(image_name.split('.')[0] + '.npy')
    # circles = run_circle_threhold(image, 8, circle_threshold=30, step=5)
    # if circles.shape[0] ==0:
    #     # f_list_test.write(fiducial_images[i])
    #     continue
    # re_r = statistics.mode(circles[:, 2])
    # circles = run_circle_threhold(image, re_r, circle_threshold=int(2.5 * re_r), step=3)
    # if circles.shape[0] ==0:
    #     # f_list_test.write(fiducial_images[i])
    #     continue
    #
    # outer_polygon, inner_polygon = get_shape_from_annotation_path(i)
    # # fp_circles = find_circles_in_polygon(circles, inner_polygon)
    # in_place_circles = find_circles_in_position(circles,inner_polygon,outer_polygon)
    # if in_place_circles.shape[0] < 400:
    #     # f_list_test.write(fiducial_images[i])
    #     continue
    auto_cnt += 1

    # np.save(image_name.split('.')[0] + '_auto.npy', circles)
    # np.save(image_name.split('.')[0] + '_shape_regulated.npy', in_place_circles)
    # continue

    # if i in cellpose_good_shape_id:
    #     f_list_train.write(fiducial_images[i].rstrip('\n') + ' 1'+'\n')
    # else:
    #     f_list_train.write(fiducial_images[i].rstrip('\n') + ' 0'+'\n')
    # continue
    # in_tissue_circles, out_tissue_circles, hard_circles, circles = get_circles_from_file(image_name)
    #
    # num_fp = np.min([fp_circles.shape[0], int(0.5* gt_circles.shape[0])])
    # fp_circles = random_select_number_elements(fp_circles, num_fp)
    # # gt_circles = random_select_percentage_elements(gt_circles, 10)
    # circles = np.vstack((fp_circles,gt_circles[:,:3]))
    # in_tissue_percentage = len(in_tissue_circles)/len(circles)


    # array1_str = np.array2string(outer_polygon, separator=',').replace('\n', '').replace(' ', '')
    # array2_str = np.array2string(inner_polygon, separator=',').replace('\n', '').replace(' ', '')


    # print(outer_polygon)
    # print(inner_polygon)
    # test = input()
    # times = max(int(in_tissue_percentage*20),1)
    # for _ in range(times):
    #     img_info = fiducial_images[i]
    #     img_info = img_info.rstrip('\n')
    #     f_list.write(f"{img_info} {array1_str} {array2_str}\n")



    mask = generate_mask(image.shape[:2], circles, -1)
    mask_bool = mask.astype(bool)

    # Create an all-zero image with the same shape as the original image
    green_mask = np.zeros_like(image)

    # Wherever the mask is True, set the color to green
    green_mask[mask_bool] = [255, 0, 0]

    # Blend the green mask with the original image
    # You can adjust the transparency by changing alpha (0 - transparent, 1 - opaque)
    alpha = 0.5
    overlay_image = cv2.addWeighted(green_mask, alpha, image, 0.5, 0)
    plt.imshow(overlay_image)
    plt.show()
    # sum = np.sum(mask)

    # print(sum/(mask.shape[0]*mask.shape[1]))
    # test = input()
    # mask = generate_weighted_mask(image.shape[:2], in_tissue_circles,out_tissue_circles,2)
    # plt.imshow(image)
    # plt.imshow(1-mask,cmap='binary',alpha=0.6)
    # plt.show()

    # save_image(mask, image_name.split('.')[0]+ '_auto_tight.png', format="L")
    # continue

    # Visualization
    # annotation_image = get_position_mask(image.shape[:2], i)
    # patches = divide_mask_into_patches(annotation_image, patch_size)
    # plt.imshow(patches)
    # plt.show()



    # patches = annotate_patches(image.shape[:2], patch_size,circles)
    # annotation_image = get_image_mask_from_annotation(image.shape[:2], patches, patch_size)
    # # continuous_patch = annotate_continuous_patches(image.shape[:2], 32,circles)
    #
    # # image = image*255
    # # image = image.astype(np.uint8)
    # # image_cp = image.copy()
    # # image = plot_circles_in_image(image,in_tissue_circles,out_tissue_circles,hard_circles,width)
    # # image = plot_circles_in_image(image,circles, width)
    # # test = input()
    #
    # # save_image(image,'/home/huifang/workspace/code/fiducial_remover/temp_result/method/hough_detection/'+str(i)+'.png')
    # # continue
    #
    # #
    # f,a = plt.subplots(1,2,figsize=(20, 10))
    # # a[0].figure(figsize=(12, 12))
    # a[0].imshow(mask)
    # # a[0].imshow(1-mask, cmap='binary', alpha=0.6)
    # a[1].imshow(patches)
    # plt.show()


    # if SAVE_FILE:
    #     save_path = annotation_path + '/masks/'
    #     image_mask = np.zeros(image.shape[:2])
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     save_name = save_path+'auto_width_2'
    #     save_mask_to_file(image_mask, circles, save_name)
    #     f_list.write(image_name[:-1]+' ' + save_name + '.png' + '\n')
print('done')
print('auto cnt is: ')
print(auto_cnt)
if SAVE_FILE:
    f_list_train.close()
    f_list_test.close()
# Divide each number by 555 to calculate percentages
# plot_pie_chart(circle_numbers)

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

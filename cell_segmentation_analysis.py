# Load the image file
# Change dir_base as needed to the directory where the downloaded example data is stored

from tifffile import imread, imwrite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
# import geopandas as gpd
import scanpy as sc

from tifffile import imread, imwrite
# from csbdeep.utils import normalize
# from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap
from PIL import Image
import tifffile
import json
import cv2
from scipy.ndimage import zoom

def plot_mask_and_save_image(title, gdf, img, cmap, output_name=None, bbox=None):
    if bbox is not None:
        # Crop the image to the bounding box
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        cropped_img = img

    # # Plot options
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #
    # # Plot the cropped image
    # axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    # axes[0].set_title(title)
    # axes[0].axis('off')

    # Create filtering polygon
    if bbox is not None:
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
        # Filter for polygons in the box
        intersects_bbox = gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = gdf[intersects_bbox]
    else:
        filtered_gdf=gdf

    # Plot the filtered polygons on the second axis
    fig, axes = plt.subplots()
    filtered_gdf.plot(cmap=cmap, ax=axes)
    axes.axis('off')
    # axes.legend(loc='upper left', bbox_to_anchor=(1.05, 1))


    # Save the plot if output_name is provided
    if output_name is not None:
        # plt.savefig(output_name, bbox_inches='tight)  # Use bbox_inches='tight' to include the legend
        fig.savefig(output_name, dpi=2400,bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
def calculate_centroid(coords):
    x_coords = [point[1] for point in coords]
    y_coords = [point[0] for point in coords]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return (centroid_y, centroid_x)


def read_labelme_json(json_file, image_shape):
    with open(json_file) as file:
        data = json.load(file)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Assuming grayscale mask
    for shape in data['shapes']:
        if shape['label'] == 'tissue' or shape['label'] == 'tissue_area':
            polygon = np.array(shape['points'])
            polygon[: ,0] = polygon[:, 0]
            polygon[:, 1] = polygon[:, 1]
            polygon = np.asarray(polygon,dtype=np.int32)
            cv2.fillPoly(mask, [polygon], color=255)
    # mask = (mask > 128)
    return mask


def count_cells_in_mask(cells, mask, mask_value):
    count = 0
    in_place_cells=[]

    for cell in cells:
        y, x = cell
        if int(y)>mask.shape[0]-1:
            y = mask.shape[0]-1
        if int(x)>mask.shape[1]-1:
            x = mask.shape[1]-1
        if mask[int(y), int(x)] == mask_value:
            count += 1
            in_place_cells.append([y,x])
    return count,np.asarray(in_place_cells)

def count_cells_in_2mask(cells, mask1,mask2, mask_value1,mask_value2):
    count = 0
    in_place_cells=[]

    for cell in cells:
        y, x = cell
        if int(y) > mask.shape[0] - 1:
            y = mask.shape[0] - 1
        if int(x) > mask.shape[1] - 1:
            x = mask.shape[1] - 1
        if mask1[int(y), int(x)] == mask_value1 and mask2[int(y), int(x)] == mask_value2:
            count += 1
            in_place_cells.append([x,y])
    return count,np.asarray(in_place_cells)


def run_stardist(filename,model):
    img = plt.imread(filename)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # plt.imshow(img)
    # plt.show()
    # Percentile normalization of the image
    # Adjust min_percentile and max_percentile as needed
    min_percentile = 5
    max_percentile = 95

    img = normalize(img, min_percentile, max_percentile)
    labels, polys = model.predict_instances_big(img, axes='YXC', block_size=4096, prob_thresh=0.01, nms_thresh=0.001,
                                                min_overlap=128, context=128, normalizer=None, n_tiles=(4, 4, 1))

    # labels, polys = model.predict_instances_big(img, axes='YXC', block_size=1028, prob_thresh=0.01, nms_thresh=0.001,
    #                                             min_overlap=128, context=128, normalizer=None, n_tiles=(4, 4, 1))

    # Creating a list to store Polygon geometries
    geometries = []
    centroids = []
    # Iterating through each nuclei in the 'polys' DataFrame
    for nuclei in range(len(polys['coord'])):
        # Extracting coordinates for the current nuclei and converting them to (y, x) format
        coords = [(y, x) for x, y in zip(polys['coord'][nuclei][0], polys['coord'][nuclei][1])]
        centroid = calculate_centroid(coords)
        centroids.append(centroid)
        # Creating a Polygon geometry from the coordinates
        geometries.append(Polygon(coords))
    centroids_array = np.array(centroids)

    # Save the centroids array to a .npy file
    np.save(filename[:-4] + '_stardist.npy', centroids_array)
    # Creating a GeoDataFrame using the Polygon geometries
    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf['id'] = [f"ID_{i + 1}" for i, _ in enumerate(gdf.index)]

    # Plot the nuclei segmentation
    # bbox=(x min,y min,x max,y max)

    # Define a single color cmap
    cmap = ListedColormap(['grey'])

    # Create Plot
    plot_mask_and_save_image(title="Region of Interest 1", gdf=gdf, cmap=cmap, img=img,
                             output_name=filename[:-4] + '_stardist.tif')


def morphological_closing(binary_mask):
    # Perform Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=9)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Perform the morphological opening
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    return opened_mask

imglist = '/home/huifang/workspace/data/imagelists/tiff_img_list.txt'
annotation_path = '/home/huifang/workspace/code/fiducial_remover/location_annotation/'
file = open(imglist)
lines = file.readlines()
# model = StarDist2D.from_pretrained('2D_versatile_he')
small_image_path = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
small_images = open(small_image_path)
small_images = small_images.readlines()

for i in range(16,20):
    print(i)
    # filename = str(i) + '.png'
    # img = plt.imread(dir_base + filename)
    line = lines[i].rstrip().split(' ')
    filename = line[0]
    annotation_id = line[1]


    # cells1 = np.load(filename[:-4] + '_recovered_stardist.npy')
    cells1 = np.load('/media/huifang/data/fiducial/tiff/recovered_tiff/'+str(i)+'_cleaned_stardist.npy')
    cells2 = np.load(filename[:-4] + '_stardist.npy')

    small_image = plt.imread(annotation_path+annotation_id+'.png')
    big_image = plt.imread(filename)
    tissue_mask = read_labelme_json(annotation_path+annotation_id+'.json', small_image.shape)
    zoom_factors = (big_image.shape[0] / tissue_mask.shape[0], big_image.shape[1] / tissue_mask.shape[1])
    tissue_mask = zoom(tissue_mask, zoom_factors, order=0)
    tissue_mask = np.transpose(tissue_mask)
    # mask = plt.imread(filename[:-4] + '_mask.tif')
    # mask = np.transpose(mask)
    mask = plt.imread(small_images[int(annotation_id)].split(' ')[0][:-4]+'_ground_truth.png')
    mask = mask*255
    mask = morphological_closing(mask)
    mask = zoom(mask,zoom_factors,order=0)
    mask = np.transpose(mask)

    # mask = morphological_closing(mask)


    # f,a = plt.subplots(2,2)
    # a[0,0].imshow(big_image)
    # a[0,1].imshow(plt.imread(filename[:-4] + '_recovered.tif'))
    # a[1,0].imshow(tissue_mask,cmap='binary')
    # a[1,0].scatter(cells2[:,1],cells2[:,0],s=0.1)
    # a[1,1].imshow(tissue_mask, cmap='binary')
    # a[1,1].scatter(cells1[:, 1], cells1[:, 0],s=0.1)
    # plt.show()


    count_recovered_cells_in_tissue, _ = count_cells_in_mask(cells1,tissue_mask,255)
    count_recovered_cells_out_tissue, _ = count_cells_in_mask(cells1, tissue_mask, 0)
    count_original_cells_in_tissue, _ = count_cells_in_mask(cells2, tissue_mask, 255)
    count_original_cells_out_tissue, _ = count_cells_in_mask(cells2, tissue_mask, 0)
    # print(count_original_cells_out_tissue)
    # print(count_recovered_cells_out_tissue)
    # print(count_original_cells_in_tissue)
    # print(count_recovered_cells_in_tissue)
    # test = input()

    in_tissue_increase = count_original_cells_in_tissue - count_recovered_cells_in_tissue
    out_tissue_increase = count_original_cells_out_tissue - count_recovered_cells_out_tissue
    # print(count_original_cells_out_tissue)
    # print(count_recovered_cells_out_tissue)
    # test = input()
    overall_increase = cells2.shape[0] - cells1.shape[0]
    print(f"Number detection out of tissue of original image: {count_original_cells_out_tissue}")
    print(f"Number detection out of tissue of recovered image: {count_recovered_cells_out_tissue}")
    print(f"Number detection in tissue of original image: {count_original_cells_in_tissue}")
    print(f"Number detection in tissue of recovered image: {count_recovered_cells_in_tissue}")
    # print(f"Increased detection out of tissue: {out_tissue_increase}")
    # print(f"Increased detection in tissue: {in_tissue_increase}")
    # test = input()
    # test = input()
    percent_diff_tissue = in_tissue_increase / count_recovered_cells_in_tissue * 100
    percent_diff_non_tissue = out_tissue_increase / count_recovered_cells_out_tissue * 100

    # Print results
    # percent_overall_increase = overall_increase / cells1.shape[0] * 100
    # print(f"Percentage increased detection: {percent_overall_increase:.2f}%")
    #
    # print(f"Percentage difference in tissue area: {percent_diff_tissue:.2f}%")
    #
    print(f"Percentage difference in non-tissue area: {percent_diff_non_tissue:.2f}%")



    # f,a = plt.subplots(1,2)
    # a[0].imshow(tissue_mask,cmap='binary')
    # a[1].imshow(mask,cmap='binary')
    # plt.show()

    # Count cells in mask areas
    count_recovered_cells_in_tissue_in_fiducial,_ = count_cells_in_2mask(cells1, tissue_mask, mask,255,255)
    count_original_cells_in_tissue_in_fiducial,_ = count_cells_in_2mask(cells2,  tissue_mask, mask,255,255)
    count_recovered_cells_in_tissue_out_fiducial, _ = count_cells_in_2mask(cells1, tissue_mask, mask, 255, 0)
    count_original_cells_in_tissue_out_fiducial, _ = count_cells_in_2mask(cells2, tissue_mask, mask, 255, 0)
    print(count_recovered_cells_in_tissue_in_fiducial)
    print(count_original_cells_in_tissue_in_fiducial)
    print(count_recovered_cells_in_tissue_out_fiducial)
    print(count_original_cells_in_tissue_out_fiducial)
    # test = input()
    # plt.scatter(cells20[:,0],cells20[:,1])
    # # plt.scatter(cells11[:,0],cells11[:,1])
    # plt.show()
    # print(count_cells1_mask1)
    # print(count_cells1_mask0)
    # print(count_cells2_mask1)
    # print(count_cells2_mask0)
    # test = input()

    # # Calculate differences
    # diff_in_tissue_in_fiducial = count_original_cells_in_tissue_in_fiducial - count_recovered_cells_in_tissue_in_fiducial
    # # Calculate percentages
    # percent_diff_mask1 = diff_in_tissue_in_fiducial / in_tissue_increase * 100
    #
    # # Print results
    #
    # # print(f"Increased detection in tissue: {in_tissue_increase}")
    # print(f"Percentage difference in in-tissue fiducial area: {percent_diff_mask1:.2f}%")


    # f, a = plt.subplots(1, 2)
    # a[0].imshow(img1)
    # a[1].imshow(img2)
    # plt.show()

    # f,a = plt.subplots(1,2)
    # a[0].imshow(img1)
    # a[1].imshow(img2)
    # plt.show()

    # test = input()
    # img1 = plt.imread('/media/huifang/data/fiducial/tiff/recovered_tiff/' + str(i) + '_cleaned_stardist.tif')
    img1 = plt.imread(filename[:-4] + '_recovered_stardist.tif')
    img2 = plt.imread(filename[:-4] + '_stardist.tif')
    # f,a = plt.subplots(2,2)
    # a[0, 0].imshow(big_image)
    # a[0, 1].imshow(plt.imread(filename[:-4]+'_recovered.tif'))
    # a[1,0].imshow(np.flipud(img1))
    # a[1,1].imshow(np.flipud(img2))
    # plt.show()

    f,a = plt.subplots(1,2)
    a[0].imshow(big_image)
    a[1].imshow(plt.imread(filename[:-4]+'_recovered.tif'))
    # a[1].imshow(plt.imread('/media/huifang/data/fiducial/tiff/recovered_tiff/' + str(i) + '_cleaned.png'))
    plt.show()
    f, a = plt.subplots(1, 2)
    a[0].imshow(img1)
    a[1].imshow(img2)
    plt.show()













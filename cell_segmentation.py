# Load the image file
# Change dir_base as needed to the directory where the downloaded example data is stored

from tifffile import imread, imwrite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import geopandas as gpd
import scanpy as sc

from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap
from PIL import Image
import tifffile
import json
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







def run_stardist(img,filename,model):
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # plt.imshow(img)
    # plt.show()
    # Percentile normalization of the image
    # Adjust min_percentile and max_percentile as needed
    min_percentile = 5
    max_percentile = 95

    img = normalize(img, min_percentile, max_percentile)
    plt.imshow(img)
    plt.show()
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
    plot_mask_and_save_image(title="Region of Interest 1", gdf=gdf, cmap=cmap, img=img)
    # plot_mask_and_save_image(title="Region of Interest 1", gdf=gdf, cmap=cmap, img=img,
    #                          output_name=filename[:-4] + '_stardist.tif')


# imglist = '/home/huifang/workspace/data/imagelists/tiff_img_list.txt'
# file = open(imglist)
# lines = file.readlines()
img_folder = '/media/huifang/data/fiducial/tiff/recovered_tiff/'
model = StarDist2D.from_pretrained('2D_versatile_he')
for i in range(16,20):
    print(i)
    img1 = plt.imread(img_folder + str(i) + '.tif')
    mask1 = plt.imread(img_folder + str(i) + '_cleaned.png')
    mask1 = mask1[:, :, 3]
    mask1[mask1 > 0] = 1

    mean_values = img1[mask1 == 0].mean(axis=0)
    img_out = np.full_like(img1, fill_value=mean_values)

    # Apply the mask: keep the values where mask1 is 1
    img_out[mask1 == 1] = img1[mask1 == 1]
    run_stardist(img_out,img_folder + str(i) + '_cleaned.png',model)

















import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from scipy.ndimage import gaussian_filter
import os
import matplotlib.colors as mcolors
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label as ndi_label
from skimage.measure import label,regionprops
from skimage.color import label2rgb
from matplotlib import cm
from matplotlib.colors import to_rgb
from scipy.ndimage import distance_transform_edt

def get_disconnected_segments(mask,minimum_tissue_size=500):


    binary_mask = mask.astype(np.uint8)
    labeled_mask, num_features = label(binary_mask, return_num=True)
    # print(f"Initial number of components: {num_features}")

    # Create a mask to store large components only
    # cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Assign new labels to each component that meets the size threshold
    regions = [r for r in regionprops(labeled_mask) if r.area >= minimum_tissue_size]

    # 2) Sort by area, descending (largest first)
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)

    # 3) Build a new mask
    cleaned_mask = np.zeros_like(labeled_mask, dtype=np.int32)
    for new_id, region in enumerate(regions_sorted, start=1):
        cleaned_mask[labeled_mask == region.label] = new_id

    # new_label = 1
    # for region in regionprops(labeled_mask):
    #     if region.area >= minimum_tissue_size:
    #         cleaned_mask[labeled_mask == region.label] = new_label
    #         new_label += 1
    return cleaned_mask
    # print(f"Number of components after removal: {np.max(cleaned_mask)}")

def get_multi_class_overlay(rgb_image,mask):
    # Your provided hex palette:
    hex_colors = ['#0000FF','#1c8041', '#1e6cb3','#501d8a',  '#e55709']
    # Convert to float RGB tuples
    custom_colors = [to_rgb(c) for c in hex_colors]
    overlay = label2rgb(mask, image=rgb_image,colors=custom_colors,bg_label=0, alpha=0.7, kind='overlay')
    overlay = (overlay * 255).astype(np.uint8)
    return overlay

# def calculate_hausdorff(m, gt):
#     """Symmetric Hausdorff distance between mask and ground truth pixels."""
#     g = gt.astype(bool)
#     pts_m = np.column_stack(np.where(m))
#     pts_g = np.column_stack(np.where(g))
#     if pts_m.size == 0 or pts_g.size == 0:
#         return np.nan
#     d1 = directed_hausdorff(pts_m, pts_g)[0]
#     d2 = directed_hausdorff(pts_g, pts_m)[0]
#     return max(d1, d2)

def calculate_hausdorff(m, gt):
    """
    Symmetric Hausdorff distance between two binary masks m and gt,
    computed via distance transforms in O(N) time rather than O(n1*n2).
    """
    # ensure boolean
    m_bool  = (m  > 0)
    gt_bool = (gt > 0)
    # if either is empty, return NaN
    if not m_bool.any() or not gt_bool.any():
        return np.nan

    # 1) Distance transform of the *background* of each mask
    #    dt_gt[x] = distance from x to nearest gt-foreground pixel
    #    dt_m[x]  = distance from x to nearest m-foreground pixel
    dt_gt = distance_transform_edt(~gt_bool)
    dt_m  = distance_transform_edt(~m_bool)

    # 2) directed distances:
    #    d(m→gt) = max over m-foreground pixels of dt_gt
    #    d(gt→m) = max over gt-foreground pixels of dt_m
    d_m_to_gt  = dt_gt[m_bool].max()
    d_gt_to_m  = dt_m [gt_bool].max()

    return float(max(d_m_to_gt, d_gt_to_m))

def calculate_component_count_diff(m, gt):
    """Absolute difference in number of connected components."""
    g = gt.astype(np.uint8)
    _, num_m = ndi_label(m)
    _, num_g = ndi_label(g)
    # num_m = m.max()
    # num_g = gt.max()
    return abs(num_m - num_g)

def calculate_perimeter_ratio(m, gt):

    g = gt.astype(np.uint8)
    m = m.astype(np.uint8)
    cnts_m, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_g, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim_m = sum(cv2.arcLength(c, True) for c in cnts_m)
    perim_g = sum(cv2.arcLength(c, True) for c in cnts_g)
    if perim_g == 0:
        return np.nan
    return perim_m / perim_g



def read_labelme_json(json_file, image_shape, label=['tissue','tissue_area']):
    with open(json_file) as file:
        data = json.load(file)
    mask = np.zeros(image_shape[:2], dtype=np.float32)  # Assuming grayscale mask
    for shape in data['shapes']:
        if shape['label'] in label:
            polygon = np.array(shape['points'])
            polygon[: ,0] = polygon[:, 0]
            polygon[:, 1] = polygon[:, 1]
            polygon = np.asarray(polygon,dtype=np.int32)
            cv2.fillPoly(mask, [polygon], color=1)
    return mask, data

def read_labelme_contour(json_file, scale, label='tissue'):
    contours = []
    with open(json_file) as file:
        data = json.load(file)
    for shape in data['shapes']:
        if shape['label'] == label:
            polygon = np.array(shape['points'])
            polygon[:, 0] *= scale[0]
            polygon[:, 1] *= scale[1]
            polygon = np.asarray(polygon, dtype=np.int32)

            # Append the first point to the end of the polygon to close the contour
            polygon = np.vstack([polygon, polygon[0]])
            polygon = gaussian_filter(polygon, sigma=0.3)

            contours.append(polygon)
    return contours


def binarize_array(array, threshold):
    """
    Binarizes a numpy array based on a threshold determined by the given percentile.

    :param array: numpy array to be binarized
    :param percentile: percentile value used to determine the threshold, defaults to 50 (median)
    :return: binarized numpy array
    """
    binary_array = (array >= threshold).astype(int)

    return binary_array

def get_iou_value(mask, gt):
    intersection = np.logical_and(mask, gt)
    union = np.logical_or(mask, gt)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def get_mask(path,h_new,w_new):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((h_new, w_new), Image.Resampling.LANCZOS)
    mask = np.array(img_pil)
    mask = mask[:,:,3]
    mask = mask * 1.0 / np.max(mask)
    mask = binarize_array(mask, 0.02)
    return mask


def get_overlay_ax(ax, image, mask):
    """
    Plots the RGB image with an overlaid mask on the provided axis.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        image (ndarray): RGB image array.
        mask (ndarray): Grayscale mask array.
    """
    # Set zero values in the mask to NaN so they won't be displayed

    # mask = mask.astype(float)  # Convert to float to allow NaN
    # mask = mask / 255
    # mask[mask < 0.2] = np.nan
    mask = (mask >= 0.2).astype(float)

    mask[mask == 0] = np.nan

    # Display the RGB image
    ax.imshow(image)
    # Define your custom color values as a list
    # color_values = ["#3f5d7d", "#ffcf4b", "#f2b134", "#f27649"]  # Example hex colors
    # colors = ["#f27649","#f2b134","#ffcf4b", "#3f5d7d"]
    # colors=['#501d8a','#1c8041','#e55709','#7e2f8c','#52bcec','#73aa43']
    # colors=['#c6d182','#eae0e9', '#e0c7e3', '#ae98b6', '#846e89']
    colors=['#d0b0c8','#b06098','#380850','blue']
    # # Create a ListedColormap
    # custom_cmap = mcolors.ListedColormap(color_values)

    # colors = ['#FEDACC', '#FD9778']
    # colors = ['#FEDACC','#FDD2C2','#FDCBB9','#FDC3B0','#FDBCA6','#FD6B48']
    # FEDACC
    # FDD2C2
    # FDCBB9
    # FDC3B0
    # FDBCA6
    # FDB49D
    # FDAD94
    # FDA58A
    # FD9E81
    # FD9778

    # Create the colormap
    # custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    custom_cmap = mcolors.ListedColormap(colors)
    # Overlay the grayscale mask with a colormap and transparency, ignoring zero (NaN) areas
    # overlay = ax.imshow(mask, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    overlay = ax.imshow(mask, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=1)

    # Add a color bar for the mask on the figure level
    cbar = plt.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mask Intensity")  # Label for the color bar

    # Remove axis ticks for a cleaner look
    ax.axis("off")


def get_overlayed_image(image_orig, mask, data, h_scale=1.0, w_scale=1.0):
    """
    - image_orig: H×W×3 array, either float in [0,1] or uint8 in [0,255]
    - mask:      H×W binary mask (0 or 255)
    - data:      LabelMe JSON dict with 'shapes' for drawing the green boundary
    - h_scale, w_scale: scaling from JSON coords → image pixels
    """
    # 1) Ensure a uint8 [0..255] image
    if image_orig.dtype != np.uint8:
        img = (np.clip(image_orig, 0, 1) * 255).astype(np.uint8)
    else:
        img = image_orig.copy()

    # 2) Create a blue overlay (RGB blue = (0,0,255))
    blue = np.zeros_like(img)
    # blue[mask == 255] = (0, 0, 255)
    blue[:,:,-1] = mask

    # 3) Alpha‑blend: result is still uint8
    alpha = 0.5
    overlay = cv2.addWeighted(img, 1 - alpha, blue, alpha, 0)

    # 4) Draw green boundary (RGB green = (0,255,0))
    for shape in data['shapes']:
        if shape.get('label') == 'tissue':
            poly = np.array(shape['points'], dtype=float)
            poly[:, 0] *= h_scale
            poly[:, 1] *= w_scale
            poly = poly.astype(np.int32)
            cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=10)

    # 5) Return uint8 image → safe for plt.imshow without clipping warnings
    return overlay


def save_overlay_to_file(masks,path):
    for name, mask in masks.items():
        if mask.shape[0] !=original_image.shape[0] or mask.shape[1] !=original_image.shape[1]:
            mask = Image.fromarray(mask)
            mask = mask.resize((original_image.shape[1], original_image.shape[0]), Image.Resampling.NEAREST)
            mask = np.asarray(mask)
        if ('Bgrm' in name) or ('gt' in name):
            overlay = get_multi_class_overlay(original_image, mask)
        else:
            overlay = get_multi_class_overlay(vispro_image, mask)
        # print(mask.max())
        # test = input()
        # print(name)
        plt.imshow(overlay)
        plt.show()
        # cv2.imwrite(path+str(sample_id)+'_'+name[1:-1]+'.png', overlay)
    # test = input()

test_image_path= '/home/huifang/workspace/data/imagelists/vispro/tissue_segmentation.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
# for line in files:
#     img = line.split(' ')[0]
visualization = True
plot = False
plot_bar=True
vispro_image_path = '/media/huifang/data/fiducial/temp_result/application/model_out/recovery/'
annotation_path = '/media/huifang/data/fiducial/annotations/location_annotation/'

mask_path_bgrm = '/media/huifang/data/fiducial/temp_result/application/bgrm-backup/'
tesla_marker_free_path = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/tesla/marker_free/'
sam_marker_free_binary_path = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/sam_binary/marker_free/'
sam_marker_free_multi_path = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/sam_multi/marker_free/'
otsu_marker_free_path = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/otsu/marker_free/'
# 2. Prepare output directory
# binary_output_dir = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/comparison/marker_free/binary/'
# multi_output_dir = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/comparison/marker_free/multi/'

tesla_path = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/tesla/'
sam_binary_path = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/sam_binary/'
sam_multi_path = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/sam_multi/'
otsu_path = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/otsu/'
# 2. Prepare output directory
binary_output_dir = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/comparison/binary/'
multi_output_dir = '/media/huifang/data/fiducial/temp_result/vispro/segmentation/comparison/multi/'

for i in range(18, len(files)):
# for i in range(17, len(files)):
    line = files[i].rstrip().split(' ')
    sample_id = int(line[0])

    print(sample_id)
    test = input()


    image_path = line[1]
    adata_path = line[2]
    scalefile_path = line[3]

    original_image = plt.imread(image_path)

    # vispro_image = plt.imread(vispro_image_path + str(sample_id) + '.png')
    vispro_image = Image.open(vispro_image_path+str(sample_id)+'.png')
    vispro_image = vispro_image.resize((original_image.shape[1], original_image.shape[0]), Image.Resampling.LANCZOS)
    vispro_image = np.array(vispro_image)

    gt_original, ground_truth_contours = read_labelme_json(annotation_path + str(sample_id) + '.json',
                                                                 original_image.shape[:2])


    try:
        mask_bgrm = get_mask(mask_path_bgrm+str(sample_id)+'original.png',original_image.shape[1],original_image.shape[0])
    except:
        mask_bgrm = get_mask(mask_path_bgrm + str(sample_id) + '_original2.png',original_image.shape[1],original_image.shape[0])

    #     results with original image
    mask_tesla1 = plt.imread(tesla_path + str(sample_id) + '_tesla1.png')
    mask_tesla2 = plt.imread(tesla_path + str(sample_id) + '_tesla2.png')
    mask_tesla3 = plt.imread(tesla_path + str(sample_id) + '_tesla3.png')
    mask_otsu = plt.imread(otsu_path + str(sample_id) + '_otsu.png')
    mask_sam_binary = plt.imread(sam_binary_path + str(sample_id) + '_sam_binary.png')


    #     results with vispro image
    mask_vispro = get_mask(mask_path_bgrm + str(sample_id) + '.png', original_image.shape[1], original_image.shape[0])
    # mask_vispro = get_mask(mask_path_bgrm + str(sample_id) + '_2.png',original_image.shape[1],original_image.shape[0])
    mask_tesla1_marker_free = plt.imread(tesla_marker_free_path + str(sample_id) + '_tesla1.png')
    mask_otsu_marker_free = plt.imread(otsu_marker_free_path + str(sample_id) + '_otsu.png')
    mask_sam_binary_marker_free = plt.imread(sam_marker_free_binary_path + str(sample_id) + '_sam_binary.png')




    binary_masks = {
        '\'gt\'':gt_original,
        # '\'Vispro\'':mask_vispro,
        # '\'Otsu\'': mask_otsu_marker_free,
        # '\'SAM\'': mask_sam_binary_marker_free,
        # '\'Tesla1\'': mask_tesla1_marker_free,
        # '\'Tesla2\'':mask_tesla2,
        # '\'Tesla3\'':mask_tesla3,
        # '\'Bgrm\'':mask_bgrm
    }



    save_overlay_to_file(binary_masks,binary_output_dir)



    # # 3. Compute IoU and save overlays
    # metrics_results = {}
    # for name, mask in binary_masks.items():
    #     if mask.shape[0] !=gt_original.shape[0] or mask.shape[1] !=gt_original.shape[1]:
    #         mask = Image.fromarray(mask)
    #         mask = mask.resize((gt_original.shape[1], gt_original.shape[0]), Image.Resampling.NEAREST)
    #         mask = np.asarray(mask)
    #
    #     iou = get_iou_value(mask, gt_original)
    #     hausdorff_dist = calculate_hausdorff(mask, gt_original)
    #     perim_ratio = calculate_perimeter_ratio(mask, gt_original)
    #     metrics_results[name] = {
    #         'iou': iou,
    #         'hausdorff': hausdorff_dist,
    #         'perimeter_ratio': perim_ratio
    #     }
        # Print a summary for this sample
    # print(f"\nMetrics results for sample {sample_id}:")
    # for name, m in metrics_results.items():
    #     print(
    #         f"  {name:<7}:( "
    #         f"{m['iou']:.4f}, "
    #         f"{m['hausdorff']:.2f}, "
    #         f"{m['perimeter_ratio']:.4f}),"
    #     )
    # print("-" * 30)



    mask_sam_multi_marker_free = plt.imread(sam_marker_free_multi_path + str(sample_id) + '_sam_multi.png')
    mask_sam_multi_marker_free = (mask_sam_multi_marker_free * 255).astype(np.uint8)

    #
    # multi_masks={'\'SAM\'' : mask_sam_multi_marker_free}
    multi_masks={}
    for name, mask in binary_masks.items():
        if ('SAM' in name):
            continue
        segmentation_masks = get_disconnected_segments(mask, 500)
        # add a new entry to the dictionary under the key `name`
        multi_masks[name] = segmentation_masks
    # # 3. Compute IoU and save overlays
    save_overlay_to_file(multi_masks, multi_output_dir)
    print('saved')
    test = input()


    # # metrics_results = {}
    # for name, mask in multi_masks.items():
    #     if mask.shape[0] !=gt_original.shape[0] or mask.shape[1] !=gt_original.shape[1]:
    #         mask = Image.fromarray(mask)
    #         mask = mask.resize((gt_original.shape[1], gt_original.shape[0]), Image.Resampling.NEAREST)
    #         mask = np.asarray(mask)
    #     iou = get_iou_value(mask, gt_original)
    #     comp_diff = calculate_component_count_diff(mask, gt_original)
    #     metrics_results[name] = {
    #         'iou': iou,
    #         'component_diff': comp_diff,
    #     }
    #     # Print a summary for this sample
    # print(f"\nMetrics results for sample {sample_id}:")
    # for name, m in metrics_results.items():
    #     print(
    #         f"  {name:<7}:( "
    #         f"{m['iou']:.4f}, "
    #         f"{m['component_diff']}), "
    #     )
    # print("-" * 30)




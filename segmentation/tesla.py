import cv2
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import TESLA as tesla
import json
from PIL import Image

def detect_contour(id,img,counts):
    # -----------------1. Detect contour using cv2-----------------
    if id==1:
        cnt = tesla.cv2_detect_contour(img, apertureSize=5, L2gradient=True)
    elif id==2:
        # -----------------2. Scan contour by x-----------------
        spots = counts.obs.loc[:, ['pixel_x', 'pixel_y', "array_x", "array_y"]]
        # shape="hexagon" for 10X Vsium, shape="square" for ST
        cnt = tesla.scan_contour(spots, scan_x=True, shape="hexagon")
    elif id==3:
        # -----------------3. Scan contour by y-----------------
        spots = counts.obs.loc[:, ['pixel_x', 'pixel_y', "array_x", "array_y"]]
        # shape="hexagon" for 10X Vsium, shape="square" for ST
        cnt = tesla.scan_contour(spots, scan_x=False, shape="hexagon")

    # Assume cnt is of shape (N, 1, 2), and you know the image dimensions
    height, width = img.shape[:2]  # or use known dimensions

    # Create a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw filled contour onto the mask
    cv2.drawContours(mask, [cnt], contourIdx=-1, color=255, thickness=cv2.FILLED)
    return cnt


def get_tesla_masks(img,adata_file,scale_file):
    adata = sc.read_h5ad(adata_file)
    if 'in_tissue' in adata.obs.columns:
        adata = adata[adata.obs['in_tissue'] == 1]
    with open(scale_file, 'r') as f:
        scalefactors = json.load(f)

    # This factor resizes from "full-resolution" coordinates to the hires image coordinates
    hires_scalef = scalefactors['tissue_hires_scalef']
    print(adata)
    test = input()

    # for most cases
    # if "pixel_x" in adata.obs and "pixel_y" in adata.obs:
    #     adata.obs['pixel_x'] = adata.obs['pixel_y'] * hires_scalef
    #     adata.obs['pixel_y'] = adata.obs['pixel_x'] * hires_scalef
    # else:
    #     adata.obs['pixel_x'] = adata.obsm['spatial'][:, 1] *hires_scalef
    #     adata.obs['pixel_y'] = adata.obsm['spatial'][:, 0] *hires_scalef
    # if "array_row" in adata.obs and "array_col" in adata.obs:
    #     adata.obs["array_y"] = adata.obs["array_col"]
    #     adata.obs["array_x"] = adata.obs["array_row"]

    if "pixel_x" in adata.obs and "pixel_y" in adata.obs:
        adata.obs['pixel_x'] = adata.obs['pixel_y'] * hires_scalef
        adata.obs['pixel_y'] = adata.obs['pixel_x'] * hires_scalef
    else:
        adata.obs['pixel_x'] = adata.obsm['spatial'][:, 1] * hires_scalef
        adata.obs['pixel_y'] = adata.obsm['spatial'][:, 0] * hires_scalef

    if "array_row" in adata.obs and "array_col" in adata.obs:
        adata.obs["array_y"] = adata.obs["array_row"]
        adata.obs["array_x"] = adata.obs["array_col"]


    img_cv = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    masks = []
    for id in [1, 2, 3]:
        cnt = detect_contour(id, img_cv, adata)
        height, width = img.shape[:2]  # or use known dimensions
        # Create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)
        # # Draw filled contour onto the mask
        cv2.drawContours(mask, [cnt], contourIdx=-1, color=255, thickness=cv2.FILLED)
        masks.append(mask)
    return masks

test_image_path= '/home/huifang/workspace/data/imagelists/vispro/tissue_segmentation.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
saving_path = "/media/huifang/data/fiducial/temp_result/vispro/segmentation/"

# for i in range(len(files)):
for i in [15]:
    line = files[i].rstrip().split(' ')
    sample_id = int(line[0])
    print(sample_id)
    # test = input()
    image_path = line[1]

    adata_path = line[2]
    scalefile_path = line[3]

    image_orig = plt.imread(image_path)

    mask_teslas = get_tesla_masks(image_orig, adata_path, scalefile_path)
    mask_tesla1 = mask_teslas[0]
    mask_tesla2 = mask_teslas[1]
    mask_tesla3 = mask_teslas[2]


    # cv2.imwrite(saving_path+str(sample_id)+'_tesla1.png', mask_tesla1)
    cv2.imwrite(saving_path + str(sample_id) + '_tesla2.png', mask_tesla2)
    cv2.imwrite(saving_path + str(sample_id) + '_tesla3.png', mask_tesla3)
    print('saved')
    test = input()


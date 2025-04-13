import os,csv,re, time
import pickle
import random
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
from scipy.sparse import issparse
import scanpy as sc
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
import TESLA as tesla
# from IPython.display import Image
from PIL import Image
import gc
print(tesla.__version__)

def detect_contour(id,img,counts,img_file):
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
    else:
        img_vispro = cv2.imread(img_file[:-4] + '_cleaned_with_bg.png')
        cnt = tesla.cv2_detect_contour(img_vispro, apertureSize=5, L2gradient=True)
    return cnt


def run_single_sample(counts_file,img_file):

    counts = sc.read(counts_file)
    img = cv2.imread(img_file)
    resize_factor = 1000 / np.min(img.shape[0:2])
    resize_width = int(img.shape[1] * resize_factor)
    resize_height = int(img.shape[0] * resize_factor)
    counts.var.index = [i.upper() for i in counts.var.index]
    counts.var_names_make_unique()
    counts.raw = counts
    sc.pp.log1p(counts)  # impute on log scale
    if issparse(counts.X): counts.X = counts.X.A.copy()




    # g="NKX2-5"
    # g="SNAP25"
    # g = "TP53"
    # g="KLK3"
    scenecence_gene = ["IL6", "IGFBP3", "EGFR", "SERPINE1", "IGFBP1", "IGFBP7", "FAS", "FGF2", "VEGFA", "CDKN1A",
                       "CDKN2A", "STAT1", "TNFRSF10C", "PARP1", "CXCL8", "IL1A", "CXCL1", "ICAM1", "CCL2", "IGFBP2",
                       "AXL", "WNT2", "HMGB2", "HMGB1", "IGFBP5", "GDF15", "MDM2", "CDKN2B", "CCNA2", "CDK1", "HELLS",
                       "FOXM1", "BUB1B", "LMNB1", "BRCA1", "IGF1", "JUN", "MIF", "TGFB1"]
    cnt_color = clr.LinearSegmentedColormap.from_list('magma', ["#000003", "#3b0f6f", "#8c2980", "#f66e5b", "#fd9f6c",
                                                                "#fbfcbf"], N=256)
    # for g in scenecence_gene:
    # for g in ["HMGB1", "MIF", "IGFBP5", "IGFBP7", "PARP1"]:
    for g in ["HMGB1"]:
        exists = g in counts.var.index
        if not exists:
            continue
        counts.obs[g] = counts.X[:, counts.var.index == g]

        # Load your 10x spatial dataset (replace with your dataset path)
        adata = sc.read_visium('/media/huifang/data/fiducial/tiff_data/151672/')  # Adjust path to point to your dataset directory

        # Plot the spatial distribution of the gene expression over the tissue image
        sc.pl.spatial(
            adata,
            color=g,
            size=1.1,  # Adjust spot size for visibility
            alpha_img=1,  # Set tissue image transparency
            img_key="hires",  # Use high-resolution image if available
            color_map="magma",  # Set the color map for gene intensity
            crop_coord=None,
            vmin=1, vmax=4,  # Set min and max for color bar (adjust based on data range)
            layer="X",  # Specify the data layer if necessary
            show=True
        )

        # fig = sc.pl.scatter(counts, alpha=1, x="pixel_y", y="pixel_x", color=g, color_map=cnt_color, show=False, size=35)
        # fig.set_aspect('equal', 'box')
        # fig.invert_yaxis()
        # plt.gcf().set_dpi(100)
        # # plt.show()
        # plt.savefig(image_file_name[:-4] + '_' + g + '_original.png', dpi=600)
        # print('saved')
        # test = input()
    # return

    for i in range(1,5):
        # print(i)
        # cnt = detect_contour(i,img,counts,img_file)
        # binary = np.zeros((img.shape[0:2]), dtype=np.uint8)
        # cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
        # # Enlarged filter
        # cnt_enlarged = tesla.scale_contour(cnt, 1.05)
        # binary_enlarged = np.zeros(img.shape[0:2])
        # cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
        # img_new = img.copy()
        # cv2.drawContours(img_new, [cnt], -1, (255), thickness=50)
        # img_new = cv2.resize(img_new, ((resize_width, resize_height)))
        # plt.imshow(img_new)
        # plt.show()
        # Set size of superpixel
        # res = 50
        # # Note, if the numer of superpixels is too large and take too long, you can increase the res to 100
        # if os.path.exists(counts_file[:-5] + "_original_imputed_"+str(i)+".h5ad"):
        #     continue
        # print(i)
        # enhanced_exp_adata = tesla.imputation(img=img, raw=counts, cnt=cnt, genes=counts.var.index.tolist(), shape="None",
        #                                       res=res, s=1, k=2, num_nbs=10)
        # enhanced_exp_adata.write_h5ad(counts_file[:-5] + "_original_imputed_"+str(i)+".h5ad")
        # print("done")
        enhanced_exp_adata = sc.read(counts_file[:-5] + "_original_imputed_"+str(i)+".h5ad", backed='r')
        # for g in scenecence_gene:
        # for g in ["HMGB1","MIF","IGFBP5","IGFBP7","PARP1"]:
        for g in ["CDKN1A", "CDKN2A"]:
            exists = g in counts.var.index
            if not exists:
                continue
            print(g)
            enhanced_exp_adata.obs[g] = enhanced_exp_adata.X[:, enhanced_exp_adata.var.index == g]
            fig = sc.pl.scatter(enhanced_exp_adata, alpha=1, x="y", y="x", color=g, color_map=cnt_color, show=False, size=10)
            fig.set_aspect('equal', 'box')
            fig.invert_yaxis()
            plt.gcf().set_dpi(100)
            plt.savefig(image_file_name[:-4] +'_'+g+ '_imputed_'+str(i)+'.png', dpi=600)
            # plt.show()
            del fig
        del enhanced_exp_adata
        gc.collect()


imglist= "/media/huifang/data/fiducial/tiff_data/data_list.txt"
file = open(imglist)
lines = file.readlines()
num_files = len(lines)
# paper figure [1,12,19,18]
for i in range(18,num_files):
    print(i)
    line = lines[i]
    line = line.rstrip().split(' ')
    adata_file_name = line[1]+'ad'
    image_file_name = line[0]
    print(image_file_name)
    # adata_file_name = "/media/huifang/data/fiducial/tiff_data/Visium_FFPE_Human_Normal_Prostate_spatial/Visium_FFPE_Human_Normal_Prostate_raw_feature_bc_matrix.h5ad"
    # image_file_name="/media/huifang/data/fiducial/tiff_data/Visium_FFPE_Human_Normal_Prostate_spatial/Visium_FFPE_Human_Normal_Prostate_image.jpg"
    # print(adata_file_name)
    # print(spatial_file_name)
    # test = input()
    # print(adata_file_name+'ad')
    # print(image_file_name[:-4] + '_cleaned_with_bg.tif')
    # if not os.path.exists(image_file_name[:-4] + '_recovered.tif'):
    #     continue

    run_single_sample(adata_file_name,image_file_name)




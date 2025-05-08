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
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import TESLA as tesla
from scipy.sparse import csr_matrix
# from IPython.display import Image
from PIL import Image
import gc
import math
from anndata import AnnData
import tifffile
print(tesla.__version__)

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
    return cnt


def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49, RGB=True):
    if RGB:
        #beta to control the range of neighbourhood when calculate grey vale for one spot
        beta_half=round(beta/2)
        g=[]
        for i in range(len(x_pixel)):
            max_x=image.shape[0]
            max_y=image.shape[1]
            nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
            g.append(np.mean(np.mean(nbs,axis=0),axis=0))
        c0, c1, c2=[], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0=np.array(c0)
        c1=np.array(c1)
        c2=np.array(c2)
        c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
    else:
        beta_half=round(beta/2)
        g=[]
        for i in range(len(x_pixel)):
            max_x=image.shape[0]
            max_y=image.shape[1]
            nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
            g.append(np.mean(nbs))
        c3=np.array(g)
    return c3

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled

def distance(t1,t2):
    sum=((t1-t2)**2).sum()
    return math.sqrt(sum)

def imputation(img, raw, cnt, genes, shape="None", res=50, s=1, k=2, num_nbs=10):
    binary=np.zeros((img.shape[0:2]), dtype=np.uint8)
    cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
    #Enlarged filter
    cnt_enlarged = scale_contour(cnt, 1.00)
    binary_enlarged = np.zeros(img.shape[0:2])
    cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)



    x_max, y_max=img.shape[0], img.shape[1]
    x_list=list(range(int(res), x_max, int(res)))
    y_list=list(range(int(res), y_max, int(res)))
    x=np.repeat(x_list,len(y_list)).tolist()
    y=y_list*len(x_list)
    sudo=pd.DataFrame({"x":x, "y": y})
    sudo=sudo[sudo.index.isin([i for i in sudo.index if (binary_enlarged[sudo.x[i], sudo.y[i]]!=0)])]
    b=res
    sudo["color"]=extract_color(x_pixel=sudo.x.tolist(), y_pixel=sudo.y.tolist(), image=img, beta=b, RGB=True)
    z_scale=np.max([np.std(sudo.x), np.std(sudo.y)])*s
    sudo["z"]=(sudo["color"]-np.mean(sudo["color"]))/np.std(sudo["color"])*z_scale
    sudo=sudo.reset_index(drop=True)
    #------------------------------------Known points---------------------------------#
    known_adata=raw[:, raw.var.index.isin(genes)]
    known_adata.obs["x"]=known_adata.obs["pixel_x"]
    known_adata.obs["y"]=known_adata.obs["pixel_y"]
    known_adata.obs["color"]=extract_color(x_pixel=known_adata.obs["pixel_x"].astype(int).tolist(), y_pixel=known_adata.obs["pixel_y"].astype(int).tolist(), image=img, beta=b, RGB=False)
    known_adata.obs["z"]=(known_adata.obs["color"]-np.mean(known_adata.obs["color"]))/np.std(known_adata.obs["color"])*z_scale
    #-----------------------Distance matrix between sudo and known points-------------#
    start_time = time.time()
    dis=np.zeros((sudo.shape[0],known_adata.shape[0]))
    x_sudo, y_sudo, z_sudo=sudo["x"].values, sudo["y"].values, sudo["z"].values
    x_known, y_known, z_known=known_adata.obs["x"].values, known_adata.obs["y"].values, known_adata.obs["z"].values
    print("Total number of sudo points: ", sudo.shape[0])
    for i in range(sudo.shape[0]):
        if i%1000==0:print("Calculating spot", i)
        cord1=np.array([x_sudo[i], y_sudo[i], z_sudo[i]])
        for j in range(known_adata.shape[0]):
            cord2=np.array([x_known[j], y_known[j], z_known[j]])
            dis[i][j]=distance(cord1, cord2)
    print("--- %s seconds ---" % (time.time() - start_time))
    dis=pd.DataFrame(dis, index=sudo.index, columns=known_adata.obs.index)
    #-------------------------Fill gene expression using nbs---------------------------#
    sudo_adata=AnnData(np.zeros((sudo.shape[0], len(genes))))
    sudo_adata.obs=sudo
    sudo_adata.var=known_adata.var
    #Impute using all spots, weighted
    for i in range(sudo_adata.shape[0]):
        if i%1000==0:print("Imputing spot", i)
        index=sudo_adata.obs.index[i]
        dis_tmp=dis.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs]
        dis_tmp=(nbs.to_numpy()+0.1)/np.min(nbs.to_numpy()+0.1) #avoid 0 distance
        if isinstance(k, int):
            weights=((1/(dis_tmp**k))/((1/(dis_tmp**k)).sum()))
        else:
            weights=np.exp(-dis_tmp)/np.sum(np.exp(-dis_tmp))
        row_index=[known_adata.obs.index.get_loc(i) for i in nbs.index]
        sudo_adata.X[i, :]=np.dot(weights, known_adata.X[row_index,:])
    return sudo_adata

def readin_tiff(tiff_path,scale):
    with tifffile.TiffFile(tiff_path) as tif:
        img_data = tif.asarray()  # Numpy array of the image

    img_data_cv = img_data.astype(np.uint8)
    # Desired new size
    new_width = img_data.shape[1] // scale
    new_height = img_data.shape[0] // scale

    # Resize using INTER_AREA (good for downsampling)
    resized_img = cv2.resize(img_data_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img


def run_single_sample(counts_file,original_img_file,vispro_img_file):
    # Set the data downsampling scale
    scale = 10
    # Set size of superpixel
    res = 20

    # Processing transcript data
    counts = sc.read(counts_file)
    counts.obs["pixel_x"] = (counts.obs["pixel_x"] /scale).astype(int)
    counts.obs["pixel_y"] = (counts.obs["pixel_y"] /scale).astype(int)
    counts.var.index = [i.upper() for i in counts.var.index]
    counts.var_names_make_unique()
    counts.raw = counts
    sc.pp.log1p(counts)  # impute on log scale
    if issparse(counts.X): counts.X = counts.X.A.copy()

    original_img = readin_tiff(original_img_file,scale)
    for i in [1,2,3]:
        img,cnt = detect_contour(i,original_img,counts)
        # Note, if the numer of superpixels is too large and take too long, you can increase the res to 100
        enhanced_exp_adata = imputation(img=img, raw=counts, cnt=cnt, genes=counts.var.index.tolist(), shape="None",
                                              res=res, s=1, k=2, num_nbs=20)
        X_dense = enhanced_exp_adata.X  # temporarily convert to dense if it's a sparse matrix
        X_dense[X_dense < 1e-2] = 0
        enhanced_exp_adata.X = csr_matrix(X_dense)
        enhanced_exp_adata.write_h5ad(counts_file[:-5] + "_res"+str(res)+"_tesla_imputed_"+str(i)+".h5ad")
        print("saved tesla expression data "+str(i))


        #visualization a gene

        cnt_color = clr.LinearSegmentedColormap.from_list('magma',
                                                          ["#000003", "#3b0f6f", "#8c2980", "#f66e5b", "#fd9f6c",
                                                           "#fbfcbf"], N=256)
        tls_genes = ["CD4", "CD8A", "CD74", "CD79A", "IL7R", "ITGAE", "CD1D", "CD3D", "CD3E", "CD8B",
                     "CD19", "CD22", "CD52", "CD79B", "CR2", "CXCL13", "CXCR5", "FCER2", "MS4A1",
                     "PDCD1", "PTGDS", "TRBC2"]
        for g in tls_genes:
            exists = g in counts.var.index
            if not exists:
                continue
            print(g)
            if issparse(enhanced_exp_adata.X):
                enhanced_exp_adata.X = enhanced_exp_adata.X.toarray()  # or use .A attribute
            gene_idx = (enhanced_exp_adata.var.index == g)
            # Slice out the column (n_obs, 1) for this gene, and make it a dense 1D array
            expr_array = enhanced_exp_adata.X[:, gene_idx].ravel()

            # Now expr_array is shape (n_obs,) which Pandas can accept
            enhanced_exp_adata.obs[g] = expr_array
            fig = sc.pl.scatter(enhanced_exp_adata, alpha=1, x="y", y="x", color=g, color_map=cnt_color, show=False,
                                size=50)
            fig.set_aspect('equal', 'box')
            fig.invert_yaxis()
            plt.gcf().set_dpi(100)
            plt.savefig(original_img_file[:-4] + '_' + g + '_imputed_' + str(i) + '.png', dpi=600)
            # plt.show()
            del fig
        del enhanced_exp_adata


# imglist= "/media/huifang/data/fiducial/tiff_data/data_list.txt"
imglist = "/home/huifang/workspace/data/imagelists/human_pilot_tesla.txt"
file = open(imglist)
lines = file.readlines()
num_files = len(lines)
# paper figure [1,12,19,18]
for i in range(1,num_files):
    print(i)
    line = lines[i]
    line = line.rstrip().split(' ')
    original_tiff_image = line[0]
    vispro_tiff_image = line[1]
    adata_file_name = line[2]
    run_single_sample(adata_file_name,original_tiff_image,vispro_tiff_image)
    print("all saved")
    test = input()




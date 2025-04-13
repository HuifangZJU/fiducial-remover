from scanpy import read_10x_h5
import pandas as pd
import matplotlib.pyplot as plt



def run_single_sample(adata_file_name,spatial_file_name):
    adata = read_10x_h5(adata_file_name)
    spatial = pd.read_csv(spatial_file_name,sep=",", header=None, na_filter=False, index_col=0)

    adata.obs["x1"] = spatial[1]
    adata.obs["array_x"] = spatial[2]
    adata.obs["array_y"] = spatial[3]
    adata.obs["pixel_x"] = spatial[4]
    adata.obs["pixel_y"] = spatial[5]

    # adata.obs["x1"]=spatial['in_tissue']
    # adata.obs["array_x"]=spatial['array_row']
    # adata.obs["array_y"]=spatial['array_col']
    # adata.obs["pixel_x"]=spatial['pxl_row_in_fullres']
    # adata.obs["pixel_y"]=spatial['pxl_col_in_fullres']
    # #Select captured samples
    #
    adata = adata[adata.obs["x1"] == 1]

    pixel_x = adata.obs["pixel_x"]
    pixel_y = adata.obs["pixel_y"]
    plt.scatter(pixel_x, pixel_y, label='pixel_x vs pixel_y', alpha=0.7)
    # Customize the plot
    plt.title('Scatter Plot of Pixel Coordinates')
    plt.xlabel('pixel_x')
    plt.ylabel('pixel_y')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    # adata=adata[adata.obs["x1"]==1]
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    adata.write_h5ad(adata_file_name + "ad")


imglist= "/media/huifang/data/fiducial/tiff_data/data_list.txt"
file = open(imglist)
lines = file.readlines()
num_files = len(lines)
for i in range(13,num_files):
    print(i)
    line = lines[i]
    line = line.rstrip().split(' ')
    adata_file_name = line[1]
    spatial_file_name = line[3]

    # print(adata_file_name)
    # print(spatial_file_name)
    # test = input()
    run_single_sample(adata_file_name,spatial_file_name)


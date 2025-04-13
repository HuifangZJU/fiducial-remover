import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.")


def load_10x_h5(dataset_dir):

    adata = sc.read_10x_h5(os.path.join(dataset_dir, "filtered_matrix.h5"))

    # Read the spatial metadata file.
    # This file can be either tissue_positions_list.csv or tissue_positions_list.txt
    # Adjust the filename if needed.
    spatial_file = os.path.join(dataset_dir, "spatial", "tissue_positions_list.csv")
    if not os.path.isfile(spatial_file):
        spatial_file = os.path.join(dataset_dir, "spatial", "tissue_positions_list.txt")

    # Read the spatial file; if it has no header, set header=None and then assign column names.
    spot_df = pd.read_csv(spatial_file, header=None)
    spot_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']

    # Optionally, filter to keep only spots that are in tissue:
    spot_df = spot_df[spot_df['in_tissue'] == 1]

    # Construct a spatial coordinates array.
    # Here we assume that x corresponds to 'pxl_col_in_fullres' and y to 'pxl_row_in_fullres'
    spatial_coords = np.vstack((spot_df['pxl_col_in_fullres'].values,
                                spot_df['pxl_row_in_fullres'].values)).T

    # Now assign it to adata.
    # Convention: adata.obsm['spatial'] should be an array of shape (n_spots, 2)
    # Depending on your downstream analyses, you might need to reorder the columns (e.g., [x, y] or [y, x]).
    adata.obsm['spatial'] = spatial_coords
    return adata

# Assume cells1 and cells2 are given lists/arrays of cell coordinates (in the same coordinate system
# as adata.obsm['spatial']). For example:
def calculate_correlations():
    imglist = '/home/huifang/workspace/data/imagelists/tiff_img_list.txt'
    annotation_path = '/media/huifang/data/fiducial/annotations/location_annotation/'
    file = open(imglist)
    lines = file.readlines()
    for i in range(0,20):
        if i==7:
            continue
        print(i)
        line = lines[i].rstrip().split(' ')
        filename = line[0]
        annotation_id = line[1]



        # Load the processed Visium h5ad file
        try:
            adata = sc.read(line[2])  # update with your file path
        except:
            adata =  load_10x_h5(line[2])
            adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()


        spot_coords = adata.obsm["spatial"]

        # plt.figure(figsize=(10, 10))
        # scatter = plt.scatter(spot_coords[:, 0], spot_coords[:, 1],
        #                       c=adata.obs["total_counts"], cmap="viridis", s=30)
        # plt.gca().invert_yaxis()  # invert y axis if the image origin is top-left
        # plt.colorbar(scatter, label="UMI Counts")
        # plt.title("Spot-Wise UMI Counts (Spatial Domain)")
        # plt.xlabel("X Coordinate")
        # plt.ylabel("Y Coordinate")
        # plt.axis("equal")
        # plt.show()


        # cells1 = np.load(filename[:-4] + '_recovered_stardist.npy')
        cells_vispro = np.load('/media/huifang/data/fiducial/tiff/recovered_tiff/'+str(i)+'_cleaned_stardist.npy')
        cells_original = np.load(filename[:-4] + '_stardist.npy')

        # Build a KDTree on the spot coordinates to assign cells quickly.
        tree = cKDTree(spot_coords)

        # Set a radius value within which a cell is considered to belong to a spot.
        # Adjust this value based on your data/scaling.
        spot_radius = 50  # example value in the same units as your spatial coordinates

        # For cells1: query the tree with an upper bound distance.
        d1, idx1 = tree.query(cells_vispro, distance_upper_bound=spot_radius)
        # For cells that did not fall into any spot, the tree returns index = len(spot_coords)
        valid1 = idx1 < len(spot_coords)
        # Count how many cells in cells1 are assigned to each spot.
        count1 = np.bincount(idx1[valid1], minlength=len(spot_coords))

        # Repeat for cells2.
        d2, idx2 = tree.query(cells_original, distance_upper_bound=spot_radius)
        valid2 = idx2 < len(spot_coords)
        count2 = np.bincount(idx2[valid2], minlength=len(spot_coords))

        # Retrieve the UMI counts from the AnnData object.
        umi_counts = adata.obs["total_counts"].values

        # Compute Pearson correlation for each cell set with the UMI counts.
        corr1, pval1 = pearsonr(umi_counts, count1)
        corr2, pval2 = pearsonr(umi_counts, count2)

        print("Correlation between spot-wise UMI counts and cell density (Vispro):",
              round(corr1, 3), "p-value:", pval1)
        print("Correlation between spot-wise UMI counts and cell density (Original):",
              round(corr2, 3), "p-value:", pval2)



vispro_corr = np.array([-0.139, 0.256, 0.287])
orig_corr   = np.array([-0.164, 0.224, 0.219])
# Compute the difference (Vispro - Original)
diff = vispro_corr - orig_corr

# Data for the three selected samples:
# labels = ["Human Cerebellum", "Human Heart", "Mouse Brain Sagittal"]
labels = ["Slice1", "Slice2", "Slice3"]

# Correlation values (Vispro and Original, sample order: 2, 4, 6)
vispro_corr = np.array([-0.139, 0.256, 0.287])
orig_corr   = np.array([-0.164, 0.224, 0.219])
# Compute the difference: (Vispro - Original)
diff = vispro_corr - orig_corr

# Choose two distinct colors from your provided base colors.
# For instance, using base_colors[2] for Vispro and base_colors[3] for Original:
color_vispro = '#c6d182'  # a greenish shade
color_original = '#ae98b6'  # a purple hue

plt.rcParams.update({'font.size': 18})
ind = np.arange(len(labels))  # x locations: array([0, 1, 2])
width = 0.35  # width of each bar

# Create a figure with two subplots; set the left panel wider than the right one.
fig, axs = plt.subplots(2, 1, figsize=(6, 9), gridspec_kw={'height_ratios': [1.5, 1]})
fig.suptitle("Spot-wise UMI-Cell Density Correlation Comparison\n(Vispro vs. Original)", fontsize=16)

# Left Subplot: Grouped Bar Chart for Correlation Values
axs[0].bar(ind - width/2, vispro_corr, width,
           color=color_vispro, label='Vispro')
axs[0].bar(ind + width/2, orig_corr, width,
           color=color_original, label='Original')
axs[0].set_xticks(ind)
axs[0].set_xticklabels(labels, ha='right')
axs[0].set_ylabel("Correlation\n(UMI vs. Cell Density)")
axs[0].set_title("Correlation by Sample")
axs[0].legend(loc='upper left', frameon=False)
axs[0].grid(axis='y', linestyle='--', alpha=0.5)

# Right Subplot: Bar Chart for the Difference (Vispro - Original)
axs[1].bar(ind, diff, width, color=color_vispro)
axs[1].set_xticks(ind)
axs[1].set_xticklabels(labels, ha='right')
axs[1].set_ylabel("Correlation Difference")
axs[1].set_title("Improvement by Vispro")
axs[1].axhline(0, color='black', lw=1)
axs[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
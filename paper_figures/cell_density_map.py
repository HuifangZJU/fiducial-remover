import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, kendalltau
import os
import pandas as pd
import warnings
import json
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
def concordance_correlation_coefficient(x, y):
    """
    Compute Lin's Concordance Correlation Coefficient (CCC) between x and y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    covariance = np.mean((x - mean_x) * (y - mean_y))
    ccc = 2 * covariance / (var_x + var_y + (mean_x - mean_y)**2)
    return ccc
def calculate_correlations():
    imglist = '/home/huifang/workspace/data/imagelists/tiff_img_list.txt'
    annotation_path = '/media/huifang/data/fiducial/annotations/location_annotation/'
    file = open(imglist)
    lines = file.readlines()
    for i in range(0,20):
        if i==7:
            continue
        line = lines[i].rstrip().split(' ')
        filename = line[0]
        annotation_id = line[1]


        # Load the processed Visium h5ad file
        try:
            adata = sc.read(line[2])  # update with your file path
        except:
            adata =  load_10x_h5(line[2])
            sc.pp.filter_cells(adata, min_counts=500)

            # Filter out genes that are detected in too few cells (adjust min_cells as needed)
            sc.pp.filter_genes(adata, min_cells=10)

            # Optionally, you can add a metric for the number of genes per spot and even filter out spots with
            # too few genes (which might indicate low quality)
            adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1).A1  # if sparse matrix
            # Example filtering: (you may adjust the threshold)
            # adata = adata[adata.obs['n_genes_by_counts'] > 200, :]

            # --- Normalization and Log Transformation ---
            # Normalize each cell (spot) so that total counts per spot equals a common scale, e.g., 10,000.
            sc.pp.normalize_total(adata, target_sum=1e4)

            # Log-transform the data to stabilize variance.
            sc.pp.log1p(adata)

            # --- Compute Total Counts (optional, for correlation purposes) ---
            # Although counts are now normalized/log-transformed, you might still want to record the raw total counts.
            adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()


        spot_coords = adata.obsm["spatial"]
        scale_file = line[3]

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
        cells_vispro = np.load('/media/huifang/data/fiducial/tiff/cleaned_tiff/'+str(i)+'_cleaned_stardist.npy')
        cells_original = np.load(filename[:-4] + '_stardist.npy')

        # Build a KDTree on the spot coordinates to assign cells quickly.
        tree = cKDTree(spot_coords)

        # Set a radius value within which a cell is considered to belong to a spot.
        # Adjust this value based on your data/scaling.

        with open(scale_file, "r") as f:
            scalefactors = json.load(f)

        # Use the 'spot_diameter_fullres' key from the JSON file
        spot_radius = scalefactors["spot_diameter_fullres"] / 2.0

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
        # Compute alternative metrics for count1
        spearman1, sp_pval1 = spearmanr(umi_counts, count1)
        kendall1, kd_pval1 = kendalltau(umi_counts, count1)
        ccc1 = concordance_correlation_coefficient(umi_counts, count1)

        # Compute alternative metrics for count2
        spearman2, sp_pval2 = spearmanr(umi_counts, count2)
        kendall2, kd_pval2 = kendalltau(umi_counts, count2)
        ccc2 = concordance_correlation_coefficient(umi_counts, count2)

        # Print the metrics
        print("Sample ID: " + str(annotation_id))

        print("For count1 (Vispro):")
        print("  Pearson: {:.3f} (p = {:.3e})".format(corr1, pval1))
        print("  Spearman: {:.3f} (p = {:.3e})".format(spearman1, sp_pval1))
        print("  Kendall Tau: {:.3f} (p = {:.3e})".format(kendall1, kd_pval1))
        print("  CCC: {:.3f}".format(ccc1))

        print("\nFor count2 (Original):")
        print("  Pearson: {:.3f} (p = {:.3e})".format(corr2, pval2))
        print("  Spearman: {:.3f} (p = {:.3e})".format(spearman2, sp_pval2))
        print("  Kendall Tau: {:.3f} (p = {:.3e})".format(kendall2, kd_pval2))
        print("  CCC: {:.3f}".format(ccc2))



def plot_significant_samples():
    vispro_corr = np.array([-0.139, 0.256, 0.287])
    orig_corr = np.array([-0.164, 0.224, 0.219])
    # Compute the difference (Vispro - Original)
    diff = vispro_corr - orig_corr

    # Data for the three selected samples:
    # labels = ["Human Cerebellum", "Human Heart", "Mouse Brain Sagittal"]
    labels = ["Slice1", "Slice2", "Slice3"]

    # Correlation values (Vispro and Original, sample order: 2, 4, 6)
    vispro_corr = np.array([-0.139, 0.256, 0.287])
    orig_corr = np.array([-0.164, 0.224, 0.219])
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
    axs[0].bar(ind - width / 2, vispro_corr, width,
               color=color_vispro, label='Vispro')
    axs[0].bar(ind + width / 2, orig_corr, width,
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


# # -------------------------------
# # Raw correlation data (for sample ids)
# # -------------------------------
# # Note: sample id 7 is missing so we have 19 samples.
# sample_ids = np.array([145, 41, 156, 20, 74, 147, 127, 67, 100, 119, 114, 138, 90, 79, 133, 102, 84, 75, 121])
#
# # Correlation values for each sample (Vispro and Original).
# corr_vispro = np.array([
#     -0.141, 0.345, -0.132, 0.326, 0.246, -0.049, 0.28,
#      0.032, 0.021, 0.011, 0.007, -0.014, 0.009, -0.034,
#     -0.035, -0.008, -0.008, 0.011, -0.017
# ])
#
# corr_original = np.array([
#     -0.14, 0.351, -0.156, 0.339, 0.217, -0.05, 0.215,
#      0.023, 0.015, 0.003, -0.01, -0.011, 0.008, -0.019,
#     -0.044, -0.009, -0.002, 0.003, -0.024
# ])
#
#
# # -------------------------------
# # Compute improvement (Vispro - Original)
# # -------------------------------
# corr_diff = corr_vispro - corr_original
#
# # -------------------------------
# # Sort data in descending order by improvement
# # -------------------------------
# sorted_indices = np.argsort(corr_diff)[::-1]
#
# # Sorted arrays:
# sel_sample_ids     = sample_ids[sorted_indices]
# sel_corr_vispro    = corr_vispro[sorted_indices]
# sel_corr_original  = corr_original[sorted_indices]
# sel_corr_diff      = corr_diff[sorted_indices]
#
# # -------------------------------
# # Plotting setup:
# # -------------------------------
# num_samples = len(sel_sample_ids)
# ind = np.arange(num_samples)    # x positions
# width = 0.35                    # width of each bar
#
# # Define two fixed colors for the two methods:
# color_vispro = '#c6d182'       # Olive/greenish for Vispro
# color_original = '#ae98b6'     # Purple hue for Original
#
# plt.rcParams.update({'font.size': 18})
#
# # Create a figure with two subplots (left panel: grouped bar, right panel: difference bar)
# fig, axs = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [3, 1]})
# # fig.suptitle("Spot-wise UMI vs. Cell Density Correlation Comparison", fontsize=16)
#
# # Left Subplot: Grouped Bar Chart for Correlations
# axs[0].bar(ind - width/2, sel_corr_vispro, width, color=color_vispro, label='Vispro')
# axs[0].bar(ind + width/2, sel_corr_original, width, color=color_original, label='Original')
#
# axs[0].set_xticks(ind)
# # Show only the image id numbers as x-axis labels
# axs[0].set_xticklabels([str(sid) for sid in sel_sample_ids], rotation=45, ha='right', fontsize=12)
# axs[0].set_xlabel("Image id")
# axs[0].set_ylabel("Correlation")
# axs[0].set_title("Spot-wise UMI vs. Cell Density Correlation Comparison")
# axs[0].legend(loc="upper center", frameon=False)
# axs[0].grid(axis='y', linestyle='--', alpha=0.5)
#
# # Right Subplot: Bar Chart for the Difference (Vispro - Original)
# axs[1].bar(ind, sel_corr_diff, width, color='green')
# axs[1].set_xticks(ind)
# axs[1].set_xticklabels([str(sid) for sid in sel_sample_ids], rotation=45, ha='right', fontsize=12)
# axs[1].set_xlabel("Image id")
# axs[1].set_ylabel("Correlation Difference")
# axs[1].set_title("Improvement by Vispro")
# axs[1].axhline(0, color='black', lw=1)
# axs[1].grid(axis='y', linestyle='--', alpha=0.5)
#
# plt.tight_layout(rect=[0, 0, 1, 0.93])
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

# -------------------------------
# Define your raw correlation data for 19 samples.
# -------------------------------
# Sample IDs (e.g., image IDs):
sample_ids = np.array([
    145, 41, 156, 20, 74, 147, 127, 67, 100,
    119, 114, 138, 90, 79, 133, 102, 84, 75, 121
])

# Correlation values for spot-wise UMI counts vs. cell density.
# For Vispro:
corr_vispro = np.array([
    -0.141, 0.345, -0.132, 0.326, 0.246, -0.049, 0.280, 0.032, 0.021,
     0.011, 0.007, -0.014, 0.009, -0.034, -0.035, -0.008, -0.008, 0.011, -0.017
])

# For Original:
corr_original = np.array([
    -0.140, 0.351, -0.156, 0.339, 0.217, -0.050, 0.215, 0.023, 0.015,
     0.003, -0.010, -0.011, 0.008, -0.019, -0.044, -0.009, -0.002, 0.003, -0.024
])

# Compute the improvement: Vispro - Original
corr_diff = corr_vispro - corr_original

# -------------------------------
# Sort the data based on improvement (descending order)
# -------------------------------
sorted_indices = np.argsort(corr_diff)[::-1]
sel_sample_ids     = sample_ids[sorted_indices]
sel_corr_vispro    = corr_vispro[sorted_indices]
sel_corr_original  = corr_original[sorted_indices]
sel_corr_diff      = corr_diff[sorted_indices]

# -------------------------------
# Plotting setup:
# -------------------------------
num_samples = len(sel_sample_ids)
ind = np.arange(num_samples)  # positions for the groups
width = 0.35  # width of each bar

# Define fixed colors
color_vispro = '#c6d182'    # Olive/greenish for Vispro
color_original = '#ae98b6'  # Purple hue for Original

plt.rcParams.update({'font.size': 16})

# Create figure with two subplots; left panel is wider than the right panel.
fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 2]})
fig.suptitle("Spot-wise UMI vs. Cell Density Correlation Comparison", fontsize=16)

# -------------------------------
# Left Subplot: Grouped Bar Chart for Correlation Values
# -------------------------------
axs[0].bar(ind - width/2, sel_corr_vispro, width, color=color_vispro, label='Vispro')
axs[0].bar(ind + width/2, sel_corr_original, width, color=color_original, label='Original')
axs[0].set_xticks(ind)
axs[0].set_xticklabels([str(sid) for sid in sel_sample_ids], rotation=45, ha='center', fontsize=12)
axs[0].set_xlabel("Image id")
axs[0].set_ylabel("Correlation")
axs[0].set_title("Correlation per Sample")
axs[0].legend(loc='best', frameon=False)
axs[0].grid(axis='y', linestyle='--', alpha=0.5)

# -------------------------------
# Right Subplot: Bar Chart for Correlation Difference (Vispro – Original)
# -------------------------------
axs[1].bar(ind, sel_corr_diff, width, color='green')
axs[1].set_xticks(ind)
axs[1].set_xticklabels([str(sid) for sid in sel_sample_ids], rotation=45, ha='center', fontsize=12)
axs[1].set_xlabel("Image id")
axs[1].set_ylabel("Correlation Difference")
axs[1].set_title("Spot-wise UMI vs. Cell Density Correlation \n Improvement by Vispro", fontsize=14)
axs[1].axhline(0, color='black', lw=1)
axs[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

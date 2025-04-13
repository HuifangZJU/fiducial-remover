# Load the image file
# Change dir_base as needed to the directory where the downloaded example data is stored

from tifffile import imread, imwrite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
# import geopandas as gpd
import scanpy as sc
from scipy.spatial import cKDTree
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
import os
from scipy.spatial import cKDTree
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import rankdata


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


def generate_spots_mask(path):
    # 1. LOAD SCALE FACTORS
    with open(path+'scalefactors_json.json', 'r') as f:
        scalefactors = json.load(f)

    # This factor resizes from "full-resolution" coordinates to the hires image coordinates
    hires_scalef = scalefactors['tissue_hires_scalef']

    # Spot diameter in "full-resolution" pixels
    spot_diameter_fullres = scalefactors['spot_diameter_fullres']

    # 2. READ SPOT POSITIONS
    # Typically, tissue_positions_list has columns like:
    #   barcode, in_tissue, array_row, array_col, pxl_col_in_fullres, pxl_row_in_fullres
    # Check your actual file format if column names differ.
    coords_df = pd.read_csv(path+'tissue_positions_list.txt', header=None)
    # Example columns if no header is present:
    coords_df.columns = [
        'barcode', 'in_tissue', 'array_row', 'array_col',
        'pxl_col_in_fullres', 'pxl_row_in_fullres'
    ]

    # 3. LOAD THE HI-RES IMAGE TO GET THE TARGET SHAPE
    hires_image = Image.open(path+'tissue_hires_image.png')
    width, height = hires_image.size

    # Create a blank mask (all zeros), same width & height as the hi-res image
    mask = np.zeros((height, width), dtype=np.uint8)

    # 4. DRAW SPOT CIRCLES ON THE MASK
    # Spot radius in hi-res coordinates:
    # We'll compute it as the diameter in fullres coords times the hires scaling factor / 2
    spot_radius_hires = int((spot_diameter_fullres * hires_scalef) / 2)

    # Loop over each spot
    for idx, row in coords_df.iterrows():
        # Only draw if in_tissue == 1 (commonly means the spot is on the tissue)
        if row['in_tissue'] == 1:
            # Convert the "fullres" x,y to hi-res x,y
            x_hires = int(row['pxl_col_in_fullres'] * hires_scalef)
            y_hires = int(row['pxl_row_in_fullres'] * hires_scalef)

            # Draw a filled circle on the mask
            cv2.circle(
                mask,
                (x_hires, y_hires),
                spot_radius_hires,
                color=255,
                thickness=-1
            )
    return mask


# def plot_cell_count_difference_in_spots(path, cells1, cells2, background=None):
#     with open(path+'scalefactors_json.json', 'r') as f:
#         scalefactors = json.load(f)
#
#     # This factor resizes from "full-resolution" coordinates to the hires image coordinates
#     hires_scalef = scalefactors['tissue_hires_scalef']
#
#     # Spot diameter in "full-resolution" pixels
#     spot_diameter_fullres = scalefactors['spot_diameter_fullres']
#
#     # 2. READ SPOT POSITIONS
#     # Typically, tissue_positions_list has columns like:
#     #   barcode, in_tissue, array_row, array_col, pxl_col_in_fullres, pxl_row_in_fullres
#     # Check your actual file format if column names differ.
#     try:
#         spot_df = pd.read_csv(path+'tissue_positions_list.txt', header=None)
#     except:
#         spot_df = pd.read_csv(path + 'tissue_positions_list.csv', header=None)
#     # Example columns if no header is present:
#     spot_df.columns = [
#         'barcode', 'in_tissue', 'array_row', 'array_col',
#         'pxl_col_in_fullres', 'pxl_row_in_fullres'
#     ]
#
#     # Calculate the spot radius in full-resolution pixels.
#     spot_radius = spot_diameter_fullres / 2.0
#     # Filter the DataFrame to include only spots inside the tissue area
#     spot_df_tissue = spot_df[spot_df['in_tissue'] == 1]
#
#     # Create an array of spot centers with ordering [y, x] (row, col) only for spots inside tissue
#     spot_coords = np.vstack((spot_df_tissue['pxl_row_in_fullres'].values,
#                              spot_df_tissue['pxl_col_in_fullres'].values)).T
#
#
#     # Build a KDTree for the spot centers
#     tree = cKDTree(spot_coords)
#
#     # Ensure the cells are numpy arrays
#     cells1 = np.array(cells1)
#     cells2 = np.array(cells2)
#
#     # Query the tree for each cell; returns the nearest spot index and the distance
#     d1, idx1 = tree.query(cells1, distance_upper_bound=spot_radius)
#     d2, idx2 = tree.query(cells2, distance_upper_bound=spot_radius)
#
#     # For cells with no valid neighbor within the radius, cKDTree returns index = tree.n
#     valid1 = idx1 < len(spot_coords)
#     valid2 = idx2 < len(spot_coords)
#
#     # Only keep valid assignments (cells that fall into at least one spot)
#     valid_idx1 = idx1[valid1]
#     valid_idx2 = idx2[valid2]
#
#     # Count cells per spot using np.bincount (ensuring counts for all spots)
#     count1 = np.bincount(valid_idx1, minlength=len(spot_coords))
#     count2 = np.bincount(valid_idx2, minlength=len(spot_coords))
#
#     # Compute the difference (cell2 count minus cell1 count)
#     count_diff = count2 - count1
#
#
#
#     # Plot the results
#     plt.figure(figsize=(10, 10))
#
#     # If a background image is provided, display it as context.
#     if background is not None:
#         plt.imshow(background, cmap='gray', alpha=0.5)
#
#     # Scatter plot the spot centers colored by the difference.
#     # Note: The ordering is [y, x] so the x coordinate is the second element.
#     scatter = plt.scatter(spot_coords[:, 0]*hires_scalef, spot_coords[:, 1]*hires_scalef, c=count_diff,
#                           cmap='coolwarm', s=30)
#     plt.colorbar(scatter, shrink=0.6, pad=0.05, label='Cell Count Difference (cell2 - cell1)')
#     plt.title("Spot-Level Difference in Cell Counts (Efficient Assignment)")
#     plt.xlabel("X coordinate")
#     plt.ylabel("Y coordinate")
#     plt.axis('equal')
#     plt.axis('off')
#     plt.gca().invert_yaxis()  # if the image origin is the top-left.
#
#     plt.savefig('./figures/5.png', dpi=600)
#     plt.show()
#
#     return spot_coords, count_diff


def plot_cell_count_difference_in_spots_2(adata, cells1, cells2, full_img, num_to_plot=20, background=None):
    """
    Visualizes spot-level cell count differences alongside gene read percentile information,
    and for each of the top spots (by absolute difference) shows a detailed zoomed-in view
    (with overlaid cell centers) together with an overview of the spot's location on the full image.

    Parameters:
      adata (AnnData): Processed 10x Visium object containing:
          - adata.obsm["spatial"]: spot coordinates (assumed to be in [x, y] order).
          - adata.obs["n_counts"]: spot-level UMI counts.
          - Optionally, adata.obs["in_tissue"] indicating tissue spots (== 1).
          - adata.uns["scalefactors"]["spot_diameter_fullres"]: spot diameter (in full-resolution pixels).
      cells1 (list/array): List/array of full-resolution cell coordinates ([x, y]) from segmentation method 1.
      cells2 (list/array): List/array of full-resolution cell coordinates ([x, y]) from segmentation method 2.
      full_img (np.ndarray): Full-resolution tissue image (e.g. from plt.imread).
      num_to_plot (int): Number of top spots (sorted by |cell count difference|) to visualize.
      background (np.ndarray, optional): Optional background image to overlay on scatter plots.

    The function:
      1. Filters spots to those in tissue (if available),
      2. Computes the gene read (UMI) percentile for each spot,
      3. Uses a KDTree to assign cells from each set to the nearest spot (within the spot radius),
      4. Computes the cell count difference per spot,
      5. Sorts spots by the absolute difference,
      6. For each top spot, creates a figure with:
            - Left Panel: Zoomed-in subregion from full_img showing the spot area with cell overlays.
            - Right Panel: Full image with the spot indicated and annotated with its gene read percentile.
    Returns:
      spot_coords (np.ndarray): Array of spot coordinates (of tissue spots, full-res).
      count_diff (np.ndarray): Array of cell count differences per spot.
      umi_percentiles (np.ndarray): Array of UMI count percentiles per spot.
    """
    # ----- Step 1. Extract spot-level data from adata -----
    if "in_tissue" in adata.obs.columns:
        tissue_mask = adata.obs["in_tissue"] == 1
        spot_coords = adata.obsm["spatial"][tissue_mask]
        umi_counts = adata.obs["total_counts"][tissue_mask].values
    else:
        spot_coords = adata.obsm["spatial"]
        umi_counts = adata.obs["total_counts"].values

    # Compute UMI percentiles (0–100) for each spot.
    umi_percentiles = rankdata(umi_counts, method='average') / float(len(umi_counts)) * 100

    # ----- Step 2. Determine spot radius using scalefactors -----
    if "scalefactors" in adata.uns and "spot_diameter_fullres" in adata.uns["scalefactors"]:
        spot_diameter_fullres = adata.uns["scalefactors"]["spot_diameter_fullres"]
    else:
        spot_diameter_fullres = 100  # default fallback value
    spot_radius = spot_diameter_fullres / 2.0

    # ----- Step 3. Assign cells to spots with a KDTree -----
    # Ensure cells coordinates are numpy arrays (assumed to be in [x, y] full-res space).
    cells1 = np.array(cells1)
    cells2 = np.array(cells2)
    tree = cKDTree(spot_coords)
    d1, idx1 = tree.query(cells1, distance_upper_bound=spot_radius)
    d2, idx2 = tree.query(cells2, distance_upper_bound=spot_radius)
    valid1 = idx1 < len(spot_coords)
    valid2 = idx2 < len(spot_coords)
    valid_idx1 = idx1[valid1]
    valid_idx2 = idx2[valid2]
    count1 = np.bincount(valid_idx1, minlength=len(spot_coords))
    count2 = np.bincount(valid_idx2, minlength=len(spot_coords))
    count_diff = count1 - count2

    # ----- Step 4. Sort spots by the absolute cell count difference -----
    sorted_indices = np.argsort(np.abs(count_diff))[::-1]  # descending order

    # ----- Step 5. For each top spot, create a two-panel figure -----
    full_height, full_width = full_img.shape[:2]

    for idx in sorted_indices[:num_to_plot]:
        # Convert spot center (spot_coords stored in [x, y])
        x_center, y_center = spot_coords[idx]
        # For full_img indexing, we use row = y and col = x.
        row_center = int(y_center)
        col_center = int(x_center)

        # Define zoomed subregion window (square window: side length = 2 * spot_radius)

        row_start = max(0, int(row_center - spot_radius))
        row_end = min(full_height, int(row_center + spot_radius))
        col_start = max(0, int(col_center - spot_radius))
        col_end = min(full_width, int(col_center + spot_radius))
        sub_img = full_img[row_start-1:row_end+1, col_start-1:col_end+1]

        # Select cells (from both sets) falling within the spot area.
        # Using squared Euclidean distance (cells coordinates in [x, y]).
        cells1_in_spot = cells1[((cells1[:, 0] - x_center) ** 2 + (cells1[:, 1] - y_center) ** 2) <= spot_radius ** 2]
        cells2_in_spot = cells2[((cells2[:, 0] - x_center) ** 2 + (cells2[:, 1] - y_center) ** 2) <= spot_radius ** 2]

        # Adjust cell coordinates relative to the subimage.
        cells1_adjusted = cells1_in_spot - np.array([col_start, row_start])
        cells2_adjusted = cells2_in_spot - np.array([col_start, row_start])

        # Create a two-panel figure.
        # fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        # axes[0].scatter(spot_coords[:, 0], spot_coords[:, 1], c=umi_percentiles, cmap='viridis', s=30)
        # # Highlight the current spot.
        # axes[0].scatter([x_center], [y_center], c='yellow', s=200, marker='o', edgecolor='red', linewidth=2)
        # axes[0].set_title(f"Current Spot UMI Percentile: {umi_percentiles[idx]:.1f}% among Total Spots")
        # axes[0].axis('equal')
        # axes[0].axis('off')
        # axes[0].invert_yaxis()
        # Define your base colors.
        base_colors = [ '#e0c7e3', '#eae0e9', '#ae98b6', '#846e89','#c6d182']

        # Create a discrete colormap from the base colors for UMI percentiles
        cmap_base = ListedColormap(base_colors)
        # Define boundaries to map UMI percentiles into 5 bins (0-20, 20-40, ... 80-100)
        boundaries = [0, 20, 40, 60, 80, 100]
        norm = BoundaryNorm(boundaries, cmap_base.N)

        # Create the subplots
        plt.rcParams.update({'font.size': 12})
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Left Panel: scatter plot of spot positions using discrete colormap
        sc = axes[0].scatter(spot_coords[:, 0], spot_coords[:, 1],
                             c=umi_percentiles, cmap=cmap_base, norm=norm, s=30)
        # Highlight the current spot.
        axes[0].scatter([x_center], [y_center], c='red', s=80, marker='o',
                        edgecolor='red', linewidth=2)
        axes[0].set_title(
            f"Current Spot UMI Percentile: {umi_percentiles[idx]:.1f}% (among all spots)"
        )
        axes[0].axis('equal')
        axes[0].axis('off')
        axes[0].invert_yaxis()

        # Add a colorbar to the left panel.
        cbar = fig.colorbar(sc, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label("UMI Percentile (%)")


        # --- Left Panel: Zoomed-in view of the spot ---
        # Display the image (use background if provided, else sub_img)
        if background is not None:
            img = axes[1].imshow(background, cmap='gray', alpha=0.5)
        else:
            img = axes[1].imshow(sub_img, cmap='gray')

        # Create a Circle patch at the desired center and radius.
        clip_circle = Circle((x_center - col_start, y_center - row_start), spot_radius,
                               transform=axes[1].transData)
        # Set the clip path for the image artist so only the circular region is shown.
        img.set_clip_path(clip_circle)

        # Optionally, draw the circle boundary (this will be drawn on top of the clipped image).
        circ = Circle((x_center - col_start, y_center - row_start), spot_radius,
                      color='red', fill=False, lw=2)
        axes[1].add_patch(circ)

        # Plot the cell detection points
        if cells1_adjusted.size > 0:
            axes[1].scatter(cells1_adjusted[:, 0], cells1_adjusted[:, 1],
                            c='blue', marker='o', s=200, label='Vispro Detection')
        if cells2_adjusted.size > 0:
            axes[1].scatter(cells2_adjusted[:, 0], cells2_adjusted[:, 1],
                            c='yellow', marker='^', s=200, label='Original Detection')

        # Position the legend if needed; adjust bbox_to_anchor for your layout.
        axes[1].legend(loc='upper left', bbox_to_anchor=(0.9, 0.95))
        axes[1].set_title(f"Detected Cells in Zoomed Spot \n (#Vispro = {count1[idx]}, #Original = {count2[idx]})")
        axes[1].axis('off')

        # --- Right Panel: Gene UMI map of spot locations ---
        # Instead of displaying the full tissue image, we show a scatter plot of all tissue spots
        # with color indicating gene UMI percentile.


        plt.tight_layout()
        plt.show()

    return spot_coords, count_diff, umi_percentiles


def plot_cell_count_difference_in_spots(path, cells1, cells2, full_img, background=None, num_to_plot=20):
    """
    Loads tissue_positions_list (either .txt or .csv) and scalefactors_json.json file from the given path.
    Only tissue spots (where in_tissue == 1) are considered.
    For each spot (using full-resolution coordinates), this function assigns cells from two sets
    (cells1 and cells2) using a KDTree, computes the cell count difference (cell2 - cell1) per spot,
    sorts the spots by the absolute difference, and for the top spots (default num_to_plot), visualizes
    the spot subregion (from full_img) with overlaid cell centers.

    Parameters:
      path (str): Directory path containing the tissue_positions_list file and scalefactors_json.json.
      cells1 (list or np.ndarray): List/array of [y, x] full-resolution cell coordinates for set 1.
      cells2 (list or np.ndarray): List/array of [y, x] full-resolution cell coordinates for set 2.
      full_img (np.ndarray): The full-resolution tissue image (as read by plt.imread, for example).
      background (np.ndarray, optional): An optional background image for the overall scatter plot.
      num_to_plot (int): Number of top spots (by absolute cell count difference) to visualize.

    Returns:
      spot_coords (np.ndarray): Array of [y, x] spot coordinates (full resolution) for tissue spots.
      count_diff (np.ndarray): Array of cell count differences (cell2 - cell1) for each spot.
    """
    # --- Load scalefactors from the JSON file ---
    scalefactors_file = os.path.join(path, "scalefactors_json.json")
    if not os.path.isfile(scalefactors_file):
        raise FileNotFoundError(f"Scalefactors file not found: {scalefactors_file}")
    with open(scalefactors_file, 'r') as f:
        scalefactors = json.load(f)
    # Use the value from the scalefactors file.
    spot_diameter_fullres = scalefactors.get("spot_diameter_fullres")
    if spot_diameter_fullres is None:
        raise ValueError("The 'spot_diameter_fullres' key was not found in the scalefactors file.")

    # --- Load tissue_positions_list file ---
    file_candidates = ["tissue_positions_list.txt", "tissue_positions_list.csv"]
    file_found = None
    for fname in file_candidates:
        full_fname = os.path.join(path, fname)
        if os.path.isfile(full_fname):
            file_found = full_fname
            break
    if file_found is None:
        raise ValueError("No tissue_positions_list file (.txt or .csv) found in the given path.")

    ext = os.path.splitext(file_found)[1].lower()
    if ext in ['.csv', '.txt']:
        spot_df = pd.read_csv(file_found, header=None)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Assume columns: 0: barcode, 1: in_tissue, 2: array_row, 3: array_col, 4: pxl_col_in_fullres, 5: pxl_row_in_fullres
    spot_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']
    # Only keep spots in tissue
    spot_df = spot_df[spot_df['in_tissue'] == 1]

    # --- Prepare spot centers (full resolution) ---
    spot_coords = np.vstack((spot_df['pxl_row_in_fullres'].values,
                             spot_df['pxl_col_in_fullres'].values)).T
    spot_radius = spot_diameter_fullres / 2.0

    # --- Count cells per spot using a KDTree for efficiency ---
    tree = cKDTree(spot_coords)
    cells1 = np.array(cells1)
    cells2 = np.array(cells2)



    # Query the tree for each cell with a distance upper bound equal to spot_radius.
    d1, idx1 = tree.query(cells1, distance_upper_bound=spot_radius)
    d2, idx2 = tree.query(cells2, distance_upper_bound=spot_radius)

    # For cells that fall outside any spot, cKDTree returns index == len(spot_coords)
    valid1 = idx1 < len(spot_coords)
    valid2 = idx2 < len(spot_coords)
    valid_idx1 = idx1[valid1]
    valid_idx2 = idx2[valid2]

    count1 = np.bincount(valid_idx1, minlength=len(spot_coords))
    count2 = np.bincount(valid_idx2, minlength=len(spot_coords))
    count_diff = count1 - count2
    # count_diff = count2

    # --- Sort spots by the absolute difference in counts ---
    sorted_indices = np.argsort(np.abs(count_diff))[::-1]  # descending order

    # --- Overall scatter plot showing differences in the full image ---
    plt.figure(figsize=(10, 10))
    if background is not None:
        plt.imshow(background, cmap='gray', alpha=0.5)
    scatter = plt.scatter(spot_coords[:, 0], spot_coords[:, 1], c=count_diff,
                          cmap='coolwarm', s=30)
    plt.colorbar(scatter, label='Cell Count Difference (cell2 - cell1)')
    plt.title("Spot-Level Difference in Cell Counts (Full Image Overview)")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.gca().invert_yaxis()  # adjust if image origin is top-left
    plt.axis('equal')
    plt.show()

    # --- Visualize top spots in full_img with overlaid cell centers ---
    for i in sorted_indices[:num_to_plot]:
        spot_center = spot_coords[i]  # [y,x]
        y_center,x_center = spot_center

        # Define a subregion around the spot (square region covering the spot)
        y0 = max(0, int(y_center - spot_radius))
        y1 = min(full_img.shape[0], int(y_center + spot_radius))
        x0 = max(0, int(x_center - spot_radius))
        x1 = min(full_img.shape[1], int(x_center + spot_radius))
        sub_img = full_img[y0:y1, x0:x1]

        # Select cells from both sets that are within the spot (using squared distance to avoid sqrt overhead)
        cells1_in_spot = cells1[((cells1[:, 0] - y_center) ** 2 + (cells1[:, 1] - x_center) ** 2) <= spot_radius ** 2]
        cells2_in_spot = cells2[((cells2[:, 0] - y_center) ** 2 + (cells2[:, 1] - x_center) ** 2) <= spot_radius ** 2]

        # Adjust cell coordinates relative to the subimage by subtracting offsets.
        cells1_adjusted = cells1_in_spot - np.array([y0, x0])
        cells2_adjusted = cells2_in_spot - np.array([y0, x0])

        # Plot the subimage with the cell overlays and the spot boundary.
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(sub_img, cmap='gray')

        # Draw a circle indicating the spot boundary (adjusting the center relative to the subimage)
        circ = Circle((x_center - x0, y_center - y0), spot_radius, color='yellow', fill=False, lw=2)
        ax.add_patch(circ)

        if cells1_adjusted.size > 0:
            ax.scatter(cells1_adjusted[:, 0], cells1_adjusted[:, 1], c='blue', marker='o', label='Cells in Vispro Image')
        if cells2_adjusted.size > 0:
            ax.scatter(cells2_adjusted[:, 0], cells2_adjusted[:, 1], c='red', marker='x', label='Cells in Original Image')

        ax.set_title(f"Spot {i} (Diff = {count_diff[i]})")
        ax.axis('off')
        plt.legend()
        plt.show()

    return spot_coords, count_diff


# Example usage:
# path = "path/to/visium_data"  # directory containing tissue_positions_list file.
# cells1 = [[100, 150], [200, 250], [300, 350]]   # Replace with your list of cell coordinates (full-res).
# cells2 = [[105, 155], [210, 260], [320, 360], [400, 450]]
# full_img = plt.imread('your_fullres_tissue_image.png')  # Read your full-resolution image.
#
# spot_coords, diff_counts = plot_cell_count_difference_in_spots(path, cells1, cells2, full_img, spot_diameter_fullres=100, num_to_plot=10)


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
        if int(y) > mask1.shape[0] - 1:
            y = mask1.shape[0] - 1
        if int(x) > mask1.shape[1] - 1:
            x = mask1.shape[1] - 1
        if mask1[int(y), int(x)] == mask_value1 and mask2[int(y), int(x)] == mask_value2:
            count += 1
            in_place_cells.append([x,y])
    return count,np.asarray(in_place_cells)


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
annotation_path = '/media/huifang/data/fiducial/annotations/location_annotation/'
file = open(imglist)
lines = file.readlines()
# model = StarDist2D.from_pretrained('2D_versatile_he')
small_image_path = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
small_images = open(small_image_path)
small_images = small_images.readlines()

for i in range(4,20):
    print(i)
    # filename = str(i) + '.png'
    # img = plt.imread(dir_base + filename)
    line = lines[i].rstrip().split(' ')
    filename = line[0]
    annotation_id = line[1]
    print(annotation_id)


    # cells1 = np.load(filename[:-4] + '_recovered_stardist.npy')
    cells1 = np.load('/media/huifang/data/fiducial/tiff/recovered_tiff/'+str(i)+'_cleaned_stardist.npy')
    cells2 = np.load(filename[:-4] + '_stardist.npy')


    small_image = plt.imread(annotation_path+annotation_id+'.png')

    big_image = plt.imread(filename)
    tissue_mask = read_labelme_json(annotation_path+annotation_id+'.json', small_image.shape)

    # spot_mask = generate_spots_mask("/media/huifang/data/registration/humanpilot/151673/spatial/")
    adata = sc.read(
        "/media/huifang/data/fiducial/data/62_STDS0000025_Heart_data_only/1/V1_Human_Heart_spatial/Human_Heart_10xvisium_processed.h5ad")  # update with your file path

    plot_cell_count_difference_in_spots_2(adata, cells1, cells2, big_image, background=None)
    # plot_cell_count_difference_in_spots("/media/huifang/data/fiducial/data/62_STDS0000025_Heart_data_only/1/V1_Human_Heart_spatial/spatial/", cells1, cells2,big_image, background=None)









    zoom_factors = (big_image.shape[0] / tissue_mask.shape[0], big_image.shape[1] / tissue_mask.shape[1])
    tissue_mask = zoom(tissue_mask, zoom_factors, order=0)
    # spot_mask = zoom(spot_mask,zoom_factors,order=0)
    tissue_mask = np.transpose(tissue_mask)
    # mask = plt.imread(filename[:-4] + '_mask.tif')
    # mask = np.transpose(mask)
    fiducial_mask = plt.imread(small_images[int(annotation_id)].split(' ')[0][:-4]+'_ground_truth.png')
    fiducial_mask = fiducial_mask*255
    fiducial_mask = morphological_closing(fiducial_mask)
    fiducial_mask = zoom(fiducial_mask,zoom_factors,order=0)
    fiducial_mask = np.transpose(fiducial_mask)

    # f, a = plt.subplots(1, 3)
    # a[0].imshow(tissue_mask, cmap='gray')
    # a[1].imshow(spot_mask, cmap='gray')
    # a[2].imshow(mask, cmap='gray')
    # plt.show()

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

    in_tissue_increase = count_recovered_cells_in_tissue - count_original_cells_in_tissue
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
    percent_diff_tissue = in_tissue_increase / count_original_cells_in_tissue * 100
    percent_diff_non_tissue = out_tissue_increase / count_original_cells_out_tissue * 100

    # Print results
    # percent_overall_increase = overall_increase / cells1.shape[0] * 100
    # print(f"Percentage increased detection: {percent_overall_increase:.2f}%")
    #
    # print(f"Percentage difference in tissue area: {percent_diff_tissue:.2f}%")
    #
    print(f"Percentage difference in tissue area: {percent_diff_tissue:.2f}%")
    test = input()
    continue



    # f,a = plt.subplots(1,2)
    # a[0].imshow(tissue_mask,cmap='binary')
    # a[1].imshow(mask,cmap='binary')
    # plt.show()

    # Count cells in mask areas
    count_recovered_cells_in_tissue_in_fiducial,_ = count_cells_in_2mask(cells1, tissue_mask, fiducial_mask,255,255)
    count_original_cells_in_tissue_in_fiducial,_ = count_cells_in_2mask(cells2,  tissue_mask, fiducial_mask,255,255)
    count_recovered_cells_in_tissue_out_fiducial, _ = count_cells_in_2mask(cells1, tissue_mask, fiducial_mask, 255, 0)
    count_original_cells_in_tissue_out_fiducial, _ = count_cells_in_2mask(cells2, tissue_mask, fiducial_mask, 255, 0)
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













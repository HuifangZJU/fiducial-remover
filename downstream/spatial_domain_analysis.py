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
import SpaGCN as spg
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def load_h5(h5_file):
    # 1) Load your backed AnnData
    adata = sc.read_h5ad(h5_file)

    # Expected:
    # AnnData object with n_obs × n_vars = 44286 × 33538 backed at '...'
    #     obs: 'x', 'y', 'color', 'z'
    #     var: 'gene_ids', 'feature_types', 'genome', 'genename'

    # 2) Move 'x' and 'y' into obsm["spatial"]
    #    obsm should be a 2D array of shape (n_obs, 2)
    coords = np.column_stack((adata.obs["x"], adata.obs["y"]))
    adata.obsm["spatial"] = coords

    # 3) (Optional) If 'color' or 'z' are not needed, you can remove them
    #    or rename them if they hold meaningful info.
    if "color" in adata.obs:
        del adata.obs["color"]

    if "z" in adata.obs:
        del adata.obs["z"]

    # 4) Set gene names as var_names if desired
    #    e.g., if "genename" is the correct identifier for each gene:
    if "genename" in adata.var.columns and not adata.var.index.equals(adata.var["genename"]):
        adata.var.index = adata.var["genename"].values
    return adata

def load_h5ad(h5ad_file):
    adata = sc.read(h5ad_file, backed=None)

    if "pixel_x" in adata.obs and "pixel_y" in adata.obs:
        coords = np.column_stack((adata.obs["pixel_x"], adata.obs["pixel_y"]))
        adata.obsm["spatial"] = coords

        # Optionally remove them from obs if you don't need them anymore
        del adata.obs["pixel_x"]
        del adata.obs["pixel_y"]

    # If you also want to remove 'x1', 'array_x', 'array_y' (or keep them if needed), do so here:
    if "x1" in adata.obs:
        del adata.obs["x1"]
    if "array_x" in adata.obs:
        del adata.obs["array_x"]
    if "array_y" in adata.obs:
        del adata.obs["array_y"]

    # 3) (Optional) Rename var.index to match the 'genename' column if desired
    if "genename" in adata.var.columns and not adata.var.index.equals(adata.var["genename"]):
        adata.var.index = adata.var["genename"].values
    return adata


import numpy as np
def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray):
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]
    ncols = 5
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    # Flatten axes so we can index easily
    axes_flat = axes.flatten()

    for dim_idx in range(n_dims):
        ax = axes_flat[dim_idx]

        # Extract the 2D patch grid for this dimension
        patch_image = patch_matrix[:, :, dim_idx]

        im = ax.imshow(
            patch_image,           # shape (out_height, out_width)
            origin='upper',        # row=0 at the top
            cmap='viridis',        # or 'gray', etc.
            aspect='auto'
        )
        ax.set_title(f"Dimension {dim_idx + 1}")
        ax.set_xlabel("Patch (x)")
        ax.set_ylabel("Patch (y)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots if n_dims < nrows*ncols
    for dim_idx in range(n_dims, nrows*ncols):
        axes_flat[dim_idx].axis("off")

    plt.tight_layout()
    plt.savefig("/media/huifang/data/fiducial/tiff_data/151509/spagcn/feature.png", dpi=300)
    test = input()

def get_gene_feature_matrix(coords: np.ndarray,
                                         reduced_data: np.ndarray,
                                         image_size=(2016, 2016),
                                         patch_size=32):
    """
    Given:
      - coords:        (N, 2) 2D positions (x, y) for each data point
      - reduced_data:  (N, D) data values at each coordinate (N points, D dims)
      - image_size:    (height, width) of the *original* large image
      - patch_size:    size of the patch to downsample into

    We'll create a patch grid of shape:
       out_height = image_size[0] // patch_size
       out_width  = image_size[1] // patch_size
      and accumulate data from reduced_data into that grid.

    Steps:
      1) For each point, compute which patch (px, py) it belongs to.
      2) Accumulate the reduced_data values into sum_array[py, px, :].
      3) Keep track of the number of points in each patch (count_array).
      4) patch_matrix = sum_array / count_array (elementwise), ignoring patches with zero count.
      5) Plot each dimension side by side using subplots.
    """
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")

    # Number of points (N) and number of dimensions (D)
    n_spots, n_dims = reduced_data.shape

    # Image size
    height, width = image_size

    # Compute the shape of the patch matrix
    out_height = height // patch_size
    out_width  = width  // patch_size

    # We'll accumulate sums in sum_array and the count of points in count_array
    sum_array = np.zeros((out_height, out_width, n_dims), dtype=float)
    count_array = np.zeros((out_height, out_width), dtype=int)

    # 1) Assign each data point to its corresponding patch
    for i in range(n_spots):
        x, y = coords[i]  # e.g., coords might be (x, y) in [0..width, 0..height]
        px = int(x) // patch_size
        py = int(y) // patch_size

        # Check if we're within valid patch bounds
        if 0 <= px < out_width and 0 <= py < out_height:
            sum_array[py, px, :] += reduced_data[i]  # Accumulate the data
            count_array[py, px] += 1

    # 2) Compute the average (or keep as sum if you prefer) for each patch
    #    We'll avoid division by zero by clipping count_array
    patch_matrix = np.zeros_like(sum_array)
    valid_mask = (count_array > 0)
    patch_matrix[valid_mask, :] = (
        sum_array[valid_mask, :] / count_array[valid_mask, np.newaxis]
    )
    print(patch_matrix.shape)
    return patch_matrix
def reduce_gene_reads(gene_reads: np.ndarray, method: str = 'pca', n_components: int = 10) -> np.ndarray:

    if not isinstance(gene_reads, np.ndarray):
        raise ValueError("gene_reads must be a NumPy array of shape (n, m).")

    if method.lower() == 'pca':
        # Principal Component Analysis
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(gene_reads)

    elif method.lower() == 'umap':
        # UMAP
        import umap.umap_ as umap
        reducer = umap.UMAP(n_components=n_components)
        reduced_data = reducer.fit_transform(gene_reads)
    else:
        raise ValueError("method must be one of ['pca', 'umap'].")

    return reduced_data

def visualize_gene_features(adata):

    sc.tl.pca(adata, n_comps=10)
    coords = adata.obsm["spatial"]
    # PCA components (n_spots, 10)
    pca_coords = adata.obsm["X_pca"]

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    for k in range(10):
        ax = axs[k // 5, k % 5]  # pick row, column in a 2x5 grid
        pc_values = pca_coords[:, k]  # PC i
        ac = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=pc_values,
            cmap="viridis",
            s=5
        )
        ax.set_title(f"PC {k + 1}")
        ax.invert_yaxis()  # Visium coordinates are often top-left origin
        plt.colorbar(ac, ax=ax)

    plt.tight_layout()
    plt.show()

# adata1 = load_h5ad("/media/huifang/data/fiducial/tiff_data/151509/filtered_matrix.h5ad")
# for i in range(5):
for i in [0,4]:
    print(i)
    # if i ==0:
    #     adata = load_h5ad("/media/huifang/data/fiducial/tiff_data/151509/filtered_matrix.h5ad")
    # else:
    #     adata = load_h5("/media/huifang/data/fiducial/tiff_data/151509/filtered_matrix_res100_imputed_"+str(i)+".h5ad")

    # adata = load_h5("/media/huifang/data/fiducial/tiff_data/151509/filtered_matrix_res100_imputed_4.h5ad")
    adata = sc.read_h5ad("/home/huifang/workspace/code/registration/paste/fused_1_2_vispro.h5ad")

    # Filter out spots (cells) with too few genes
    sc.pp.filter_cells(adata, min_genes=100)

    # Filter out genes detected in too few spots
    # sc.pp.filter_cells(adata, max_genes=5000)
    sc.pp.filter_genes(adata, min_cells=3)
    # (C) Normalize & log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    # 7. Scale and cap values at 10
    sc.pp.scale(adata, max_value=10)
    coords = adata.obsm["spatial"]  # shape (n_obs, 2)
    # visualize_gene_features(adata)
    # continue




    # 2. Build a distance matrix (adj) using Euclidean distances
    #    Because your SpaGCN code treats 'adj' as a distance matrix and applies Gaussian transform.
    dist_mat = cdist(coords, coords)  # shape (n_obs, n_obs)

    # 3. Create the SpaGCN model
    model = spg.SpaGCN()

    # 4. Set the length-scale 'l' before training.
    #    This is crucial because the code checks if self.l is None and raises an error otherwise.
    model.set_l(200)
    # Adjust '200' to a suitable length scale for your data.
    # It should be in the same coordinate units as coords.

    # 5. Train the model
    #    Choose 'init="kmeans"' if you know the desired cluster count (e.g. n_clusters=6).
    #    Alternatively, use 'init="louvain"', which requires a resolution parameter (res=0.4, etc.)
    model.train(
        adata=adata,
        adj=dist_mat,
        num_pcs=10,
        lr=0.005,
        max_epochs=2000,
        init="kmeans",   # or "louvain"
        n_clusters=5,    # needed for kmeans
        n_neighbors=10,
        tol=1e-5,
        init_spa=True
    )


    # 6. Predict cluster labels and probabilities
    y_pred, prob = model.predict()

    # 7. Store the resulting domain labels in adata.obs
    adata.obs["spagcn_clusters"] = y_pred
    # Now do your plotting
    plt.scatter(
        adata.obsm["spatial"][:, 0],
        adata.obsm["spatial"][:, 1],
        c=adata.obs["spagcn_clusters"],
        cmap="tab20",
        s=10
    )

    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')  # Make axes equal

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("SpaGCN Clusters")
    plt.show()
    # plt.savefig("/media/huifang/data/fiducial/tiff_data/151509/spagcn/test"+str(i)+".png", dpi=300)
    # plt.close()
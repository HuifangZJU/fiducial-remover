import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

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


for i in [4]:
    # print(i)
    # if i ==0:
    #     adata = load_h5ad("/media/huifang/data/fiducial/tiff_data/151509/filtered_matrix.h5ad")
    # else:
    # adata = load_h5ad("/media/huifang/data/fiducial/tiff_data/Visium_FFPE_Human_Cervical_Cancer_spatial/Visium_FFPE_Human_Cervical_Cancer_raw_feature_bc_matrix_res100_imputed_"+str(i)+".h5ad")
    adata = load_h5ad(
        "/media/huifang/data/fiducial/tiff_data/Visium_Human_Breast_Cancer_spatial/Visium_Human_Breast_Cancer_raw_feature_bc_matrix_res20_imputed_4.h5ad")

    # Force the index to plain object/string type, not categorical
    adata.var.index = adata.var.index.astype(str)

    # Now this won't fail because it's no longer categorical
    adata.var_names_make_unique()



    coords = np.column_stack([adata.obs["x"], adata.obs["y"]])
    adata.obsm["spatial"] = coords

    # 2) (Optional) Remove extra columns if not needed
    for col in ["color", "z"]:
        if col in adata.obs:
            del adata.obs[col]

    # # 3) (Optional) If you want 'genename' to become your var.index:
    # if "genename" in adata.var.columns and not adata.var.index.equals(adata.var["genename"]):
    #     adata.var.index = adata.var["genename"]

    # adata = sc.read_h5ad("/home/huifang/workspace/code/registration/data/DLPFC/151509_preprocessed.h5")
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

    marker_sets = {
        "TLS": [
            "CD4", "CD8A", "CD74", "CD79A", "IL7R", "ITGAE", "CD1D", "CD3D", "CD3E", "CD8B",
            "CD19", "CD22", "CD52", "CD79B", "CR2", "CXCL13", "CXCR5", "FCER2", "MS4A1",
            "PDCD1", "PTGDS", "TRBC2"
        ],  # Tertiary Lymphoid Structure (your list)

        "T_cells": [
            "CD3D", "CD3E", "CD2", "CD5", "CD28", "TRBC2"
        ],  # Generic T-cell markers

        "B_cells": [
            "CD19", "MS4A1", "CD79A", "CD79B", "CD22", "CR2"
        ],  # Generic B-cell markers

        "Myeloid": [
            "CD14", "CD68", "S100A8", "S100A9", "FCGR3A"  # Monocytes/macrophages
        ],

        # Add more sets as needed, e.g. "NK_cells", "Treg", etc.
    }

    # 2. Suppose you already have an AnnData, called adata, with .obsm["spatial"] coords
    coords = adata.obsm["spatial"]

    # 3. For each marker set, compute a score and store it in adata.obs
    for set_name, genes in marker_sets.items():
        # Filter out genes that aren't in var_names
        valid_genes = [g for g in genes if g in adata.var_names]
        if not valid_genes:
            print(f"No valid genes found for set {set_name}, skipping.")
            continue

        sc.tl.score_genes(
            adata,
            gene_list=valid_genes,
            score_name=f"{set_name}_Score",
            use_raw=False  # or True, if your .raw is up to date
        )

    # 4. Plot each score in a separate subplot
    n_sets = len(marker_sets)  # total number of marker sets defined
    fig, axs = plt.subplots(1, n_sets, figsize=(5 * n_sets, 5), squeeze=False)
    # axs is 2D even if we have 1 row, so we can index axs[0, i]

    for i, (set_name, genes) in enumerate(marker_sets.items()):
        score_col = f"{set_name}_Score"
        if score_col not in adata.obs.columns:
            continue  # skip sets that had no valid genes

        ax = axs[0, i]
        scores = adata.obs[score_col]

        scat = ax.scatter(
            coords[:, 1],  # Y axis first if you want (y, x) for Visium
            coords[:, 0],
            c=scores,
            s=30,
            cmap='viridis',
            alpha=0.8,
            edgecolor='none'
        )

        ax.invert_yaxis()  # Common for Visium coordinates
        ax.set_aspect('equal', 'box')
        ax.set_title(f"{set_name} Score", fontsize=12)
        ax.set_xlabel("Spatial Y")
        ax.set_ylabel("Spatial X")

        cbar = fig.colorbar(scat, ax=ax, shrink=0.8)
        cbar.set_label(f"{set_name}_Score", fontsize=10)

    plt.tight_layout()
    plt.show()
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt



def load_h5ad(h5ad_file):
    adata = sc.read(h5ad_file, backed=None)

    if "pixel_x" in adata.obs and "pixel_y" in adata.obs:
        coords = np.column_stack((adata.obs["pixel_x"], adata.obs["pixel_y"]))
        adata.obsm["spatial"] = coords

        # Optionally remove them from obs if you don't need them anymore
        del adata.obs["pixel_x"]
        del adata.obs["pixel_y"]
    elif "x" in adata.obs and "y" in adata.obs:
        coords = np.column_stack((adata.obs["x"], adata.obs["y"]))
        adata.obsm["spatial"] = coords

        # Optionally remove them from obs if you don't need them anymore
        del adata.obs["x"]
        del adata.obs["y"]

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



def compare_global_metrics(original_adata, imputed_adata, method_name):
    """
    Compute global metrics (mean gene expression correlation and RMSE) using the full dataset.
    This includes all spots, even if some (like those from Tesla) are off-tissue.
    """
    # Force var names to be strings and unique
    for ad in [original_adata, imputed_adata]:
        ad.var.index = ad.var.index.astype(str)
        ad.var_names_make_unique()

    # Intersect the gene sets (common genes)
    common_vars = original_adata.var_names.intersection(imputed_adata.var_names)
    orig_sub = original_adata[:, common_vars].copy()
    imp_sub = imputed_adata[:, common_vars].copy()

    # Convert to dense arrays if necessary
    X_orig = orig_sub.X.toarray() if not isinstance(orig_sub.X, np.ndarray) else orig_sub.X
    X_imp = imp_sub.X.toarray() if not isinstance(imp_sub.X, np.ndarray) else imp_sub.X

    # Compute the mean expression per gene across all spots (global profile)
    mean_orig = X_orig.mean(axis=0)
    mean_imp = X_imp.mean(axis=0)

    # Compute Pearson correlation of the mean gene expression profiles
    corr = np.corrcoef(mean_orig, mean_imp)[0, 1]
    diff = mean_imp - mean_orig
    rmse_mean = np.sqrt(np.mean(diff ** 2))

    print(f"[Global stats: {method_name}] "
          f"#genes={len(common_vars)}, corr(mean expr)={corr:.4f}, RMSE(mean expr)={rmse_mean:.4f}")

    return {
        "method": method_name,
        "global_gene_corr": corr,
        "global_gene_rmse": rmse_mean
    }

########################################
# 2) SPOT-LEVEL METRICS (NN ALIGNMENT)
########################################
def nearest_neighbor_mapping(adata_ref, adata_query, max_dist=20.0):
    """
    For each spot in adata_ref, find nearest spot in adata_query by (x,y).
    Return dict: ref_obs_name -> query_obs_name, for pairs within max_dist.
    """
    coords_ref = adata_ref.obsm["spatial"]
    coords_query = adata_query.obsm["spatial"]

    tree = cKDTree(coords_query)
    dist, idx = tree.query(coords_ref)  # for each ref spot, nearest query spot
    mapping = {}
    for i, ref_name in enumerate(adata_ref.obs_names):
        if dist[i] <= max_dist:
            query_name = adata_query.obs_names[idx[i]]
            mapping[ref_name] = query_name
    return mapping


def compare_spotlevel_metrics_nn(original_adata, imputed_adata, method_name, max_dist=20.0, plot_spatial=True):
    """
    Compare original and imputed AnnData objects on a spot-level basis.

    1) Match each original spot to the nearest neighbor in imputed_adata (within max_dist).
    2) Intersect gene sets and compute spot-wise metrics:
         - RMSE over all matched spots
         - Gene-wise correlation (averaged)
         - Spot-wise correlation (averaged) per matched spot.
    3) Optionally plot a spatial scatter plot of the matched original spots colored by their spot-wise correlation.

    Returns a dict with computed metrics and the array of per-spot correlations.
    """
    # Ensure both datasets have a spatial coordinate field and unique var names
    for ad in [original_adata, imputed_adata]:
        if "spatial" not in ad.obsm:
            raise ValueError("Both AnnData objects must have .obsm['spatial'].")
        ad.var.index = ad.var.index.astype(str)
        ad.var_names_make_unique()

    # Get nearest-neighbor mapping from original to imputed
    mapping = nearest_neighbor_mapping(original_adata, imputed_adata, max_dist=max_dist)
    if len(mapping) == 0:
        print(f"No matched spots found within max_dist={max_dist} for {method_name}.")
        return {
            "method": method_name,
            "matched_spots": 0,
            "spot_rmse": np.nan,
            "mean_gene_corr": np.nan,
            "mean_spot_corr": np.nan,
            "spot_corrs": np.array([]),
        }

    matched_orig_spots = list(mapping.keys())
    matched_imp_spots = list(mapping.values())

    orig_sub = original_adata[matched_orig_spots, :].copy()
    imp_sub = imputed_adata[matched_imp_spots, :].copy()

    # Intersect gene sets
    common_vars = orig_sub.var_names.intersection(imp_sub.var_names)
    orig_sub = orig_sub[:, common_vars].copy()
    imp_sub = imp_sub[:, common_vars].copy()

    # Convert X to dense arrays if necessary
    X_orig = orig_sub.X.toarray() if not isinstance(orig_sub.X, np.ndarray) else orig_sub.X
    X_imp = imp_sub.X.toarray() if not isinstance(imp_sub.X, np.ndarray) else imp_sub.X

    diff = X_imp - X_orig
    spot_rmse = np.sqrt(np.mean(diff ** 2))

    n_spots, n_genes = X_orig.shape

    # Compute gene-wise correlations (across spots) for each gene
    gene_corrs = np.zeros(n_genes)
    for g in range(n_genes):
        gene_corrs[g] = np.corrcoef(X_orig[:, g], X_imp[:, g])[0, 1]
    mean_gene_corr = np.nanmean(gene_corrs)

    # Compute spot-wise correlations (across genes) for each matched spot
    spot_corrs = np.zeros(n_spots)
    for s in range(n_spots):
        spot_corrs[s] = np.corrcoef(X_orig[s, :], X_imp[s, :])[0, 1]
    mean_spot_corr = np.nanmean(spot_corrs)

    print(f"[Spot-level: {method_name}] matched_spots={n_spots}, #genes={n_genes}, "
          f"RMSE={spot_rmse:.4f}, MeanGeneCorr={mean_gene_corr:.4f}, MeanSpotCorr={mean_spot_corr:.4f}")

    if plot_spatial:
        # Get spatial coordinates of the matched original spots
        coords_orig = orig_sub.obsm["spatial"]
        plt.figure(figsize=(6, 6))
        scat = plt.scatter(coords_orig[:, 1], coords_orig[:, 0], c=spot_corrs, cmap="viridis",
                           s=30, alpha=0.8, edgecolor='k')
        plt.gca().invert_yaxis()  # common for Visium data
        cbar = plt.colorbar(scat)
        cbar.set_label("Spot-wise correlation", fontsize=12)
        plt.title(f"Spot-wise Correlation Map: {method_name}", fontsize=14)
        plt.xlabel("Spatial X", fontsize=12)
        plt.ylabel("Spatial Y", fontsize=12)
        plt.tight_layout()
        plt.show()

    # return {
    #     "method": method_name,
    #     "matched_spots": n_spots,
    #     "spot_rmse": spot_rmse,
    #     "mean_gene_corr": mean_gene_corr,
    #     "mean_spot_corr": mean_spot_corr,
    #     "spot_corrs": spot_corrs
    # }

    return {
            "method": method_name,
            "mean_gene_corr": mean_gene_corr,
            "mean_spot_corr": mean_spot_corr,
            "spot_corrs": spot_corrs
        }

########################################
# 3) WRAPPING IT TOGETHER
########################################
def compare_imputations(original_adata, imputed_adatas, method_names, max_dist=20.0):
    """
    For each imputed_adata:
      - Compare global (full-data) metrics
      - Compare spot-level metrics (NN alignment)
    Return a combined DataFrame or two DataFrames.
    """
    global_results = []
    spot_results = []

    for ad_imp, name in zip(imputed_adatas, method_names):
        # A) Global
        res_g = compare_global_metrics(original_adata, ad_imp, name)
        global_results.append(res_g)

        # B) Spot-level
        res_s = compare_spotlevel_metrics_nn(original_adata, ad_imp, name, max_dist)
        spot_results.append(res_s)

    df_global = pd.DataFrame(global_results)
    df_spot = pd.DataFrame(spot_results)

    return df_global, df_spot





import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def nearest_neighbor_mapping(adata_ref, adata_query, max_dist=30.0):
    """
    For each spot in adata_ref, find the nearest spot in adata_query based on spatial (x,y) distance.
    Returns a dictionary: ref_obs_name -> query_obs_name for pairs within max_dist.
    """
    coords_ref = adata_ref.obsm["spatial"]
    coords_query = adata_query.obsm["spatial"]
    tree = cKDTree(coords_query)
    dist, idx = tree.query(coords_ref)
    mapping = {}
    for i, ref_name in enumerate(adata_ref.obs_names):
        if dist[i] <= max_dist:
            mapping[ref_name] = adata_query.obs_names[idx[i]]
    return mapping

def compute_gene_correlations_per_method(original_adata, imputed_adata, max_dist=30.0):
    """
    Computes a dictionary of gene-wise spot-level correlations between original and imputed data.
    Uses nearest-neighbor mapping (with max_dist) to align spots.
    Returns: {gene: correlation}
    """
    mapping = nearest_neighbor_mapping(original_adata, imputed_adata, max_dist=max_dist)
    if len(mapping) == 0:
        print("No matched spots found!")
        return {}
    matched_orig_spots = list(mapping.keys())
    matched_imp_spots  = list(mapping.values())
    orig_sub = original_adata[matched_orig_spots, :].copy()
    imp_sub  = imputed_adata[matched_imp_spots, :].copy()
    common_vars = orig_sub.var_names.intersection(imp_sub.var_names)
    orig_sub = orig_sub[:, common_vars].copy()
    imp_sub  = imp_sub[:, common_vars].copy()
    X_orig = orig_sub.X.toarray() if not isinstance(orig_sub.X, np.ndarray) else orig_sub.X
    X_imp  = imp_sub.X.toarray()  if not isinstance(imp_sub.X, np.ndarray) else imp_sub.X

    gene_corrs = {}
    for i, gene in enumerate(common_vars):
        # Compute the correlation between original and imputed expression across spots
        corr = np.corrcoef(X_orig[:, i], X_imp[:, i])[0, 1]
        gene_corrs[gene] = corr
    return gene_corrs


def plot_gene_spatial_comparison(adata_list, method_names, gene, title=None):
    """
    Plot a spatial comparison of a given gene's expression across multiple methods.

    Parameters:
    - adata_list: list of AnnData objects (one per method). Each must have .obsm["spatial"].
    - method_names: list of method names (strings) for labeling each subplot.
    - gene: gene name to visualize.
    - title: (optional) overall title for the figure.

    The function computes the overall min and max expression for the gene among all datasets
    and uses that as vmin/vmax for a common color scale.
    """
    expr_list = []
    for ad in adata_list:
        # Extract expression of gene
        expr = ad[:, gene].X
        if not isinstance(expr, np.ndarray):
            expr = expr.toarray().ravel()
        else:
            expr = expr.ravel()
        expr_list.append(expr)

    # Compute overall min and max expression across methods for this gene
    vmin = min(np.min(e) for e in expr_list)
    vmax = max(np.max(e) for e in expr_list)

    n_methods = len(adata_list)
    fig, axs = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), squeeze=False)
    axs = axs[0]  # get the row of axes

    for i, (ad, method, expr) in enumerate(zip(adata_list, method_names, expr_list)):
        coords = ad.obsm["spatial"]
        sc_plot = axs[i].scatter(
            coords[:, 0],  # assuming x-coordinate is in column 0
            coords[:, 1],  # y-coordinate in column 1
            c=expr,
            cmap="viridis",
            s=50,
            edgecolor="k",
            alpha=0.8,
            vmin=vmin,
            vmax=vmax
        )
        axs[i].invert_yaxis()  # if needed (e.g., for Visium)
        axs[i].set_title(method, fontsize=12)
        axs[i].set_xlabel("Spatial X", fontsize=10)
        axs[i].set_ylabel("Spatial Y", fontsize=10)

    fig.suptitle(title if title else gene, fontsize=16)
    # Add one common colorbar on the right
    cbar = fig.colorbar(sc_plot, ax=axs, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label(f"{gene} expression", fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()

def filter_genes_by_variance(adata, genes, threshold=0.1):
    """Return a subset of genes that have variance above the threshold in the original data."""
    # Extract expression for the selected genes
    X = adata[:, genes].X.toarray() if not isinstance(adata[:, genes].X, np.ndarray) else adata[:, genes].X
    variances = X.var(axis=0)
    filtered_genes = [gene for gene, var in zip(genes, variances) if var >= threshold]
    return filtered_genes

########################################
# Example usage:
if __name__ == "__main__":
    # Suppose:
    original_adata = load_h5ad("/media/huifang/data/fiducial/tiff_data/Visium_FFPE_Human_Cervical_Cancer_spatial/Visium_FFPE_Human_Cervical_Cancer_raw_feature_bc_matrix.h5ad")
    original_adata.var_names = original_adata.var_names.astype(str)
    # Now safely make them unique
    original_adata.var_names_make_unique()



    methodA_adata = load_h5ad(
        "/media/huifang/data/fiducial/tiff_data/Visium_FFPE_Human_Cervical_Cancer_spatial/Visium_FFPE_Human_Cervical_Cancer_raw_feature_bc_matrix_res100_imputed_1.h5ad")
    # Ensure var names are plain strings
    methodA_adata.var_names = methodA_adata.var_names.astype(str)
    # Now safely make them unique
    methodA_adata.var_names_make_unique()

    methodB_adata = load_h5ad(
        "/media/huifang/data/fiducial/tiff_data/Visium_FFPE_Human_Cervical_Cancer_spatial/Visium_FFPE_Human_Cervical_Cancer_raw_feature_bc_matrix_res100_imputed_2.h5ad")
    # Ensure var names are plain strings
    methodB_adata.var_names = methodB_adata.var_names.astype(str)
    # Now safely make them unique
    methodB_adata.var_names_make_unique()

    methodC_adata = load_h5ad(
        "/media/huifang/data/fiducial/tiff_data/Visium_FFPE_Human_Cervical_Cancer_spatial/Visium_FFPE_Human_Cervical_Cancer_raw_feature_bc_matrix_res100_imputed_3.h5ad")
    # Ensure var names are plain strings
    methodC_adata.var_names = methodC_adata.var_names.astype(str)
    # Now safely make them unique
    methodC_adata.var_names_make_unique()

    methodD_adata = load_h5ad(
        "/media/huifang/data/fiducial/tiff_data/Visium_FFPE_Human_Cervical_Cancer_spatial/Visium_FFPE_Human_Cervical_Cancer_raw_feature_bc_matrix_res100_imputed_4.h5ad")
    # Ensure var names are plain strings
    methodD_adata.var_names = methodD_adata.var_names.astype(str)
    # Now safely make them unique
    methodD_adata.var_names_make_unique()

    # # Compute per-gene correlations for each method.
    # # (Assume original_adata, vispro_adata, tesla1_adata, tesla2_adata, tesla3_adata are loaded and have .obsm["spatial"].)
    # gene_corr_vispro = compute_gene_correlations_per_method(original_adata, methodA_adata, max_dist=100.0)
    # gene_corr_tesla1 = compute_gene_correlations_per_method(original_adata, methodB_adata, max_dist=100.0)
    # gene_corr_tesla2 = compute_gene_correlations_per_method(original_adata, methodC_adata, max_dist=100.0)
    # gene_corr_tesla3 = compute_gene_correlations_per_method(original_adata, methodD_adata, max_dist=100.0)
    #
    # # Get the common genes (as before)
    # common_genes = list(gene_corr_vispro.keys())
    #
    # # Filter common_genes based on variance in the original data (assuming original_adata is available)
    # filtered_common_genes = filter_genes_by_variance(original_adata, common_genes, threshold=0.1)
    # print("Number of genes after variance filtering:", len(filtered_common_genes))
    #
    # # Now compute the average Tesla correlations over the filtered genes:
    # avg_tesla_corr = {}
    # for gene in filtered_common_genes:
    #     vals = [
    #         gene_corr_tesla1.get(gene, np.nan),
    #         gene_corr_tesla2.get(gene, np.nan),
    #         gene_corr_tesla3.get(gene, np.nan)
    #     ]
    #     avg_tesla_corr[gene] = np.nanmean(vals)
    #
    # # Then you can compare these to the Vispro correlations:
    # diff_corr = {gene: gene_corr_vispro[gene] - avg_tesla_corr[gene] for gene in filtered_common_genes}
    #
    # # And select top genes where Vispro outperforms Tesla
    # top_genes = sorted(diff_corr.items(), key=lambda x: x[1], reverse=True)[:10]
    # selected_genes = [gene for gene, diff_val in top_genes if diff_val > 0]
    # print("Selected gene panel (Vispro outperforms Tesla):", selected_genes)
    # test = input()
    #
    #
    #
    # # Visualize the selected gene panel on the Vispro imputed data (which is performing best)
    # adata_list = [methodA_adata,methodB_adata,methodC_adata,methodD_adata]
    # method_names = ["Tesla1", "Tesla2", "Tesla3", "Vispro"]
    #
    #
    # for gene in selected_genes:
    #     plot_gene_spatial_comparison(adata_list, method_names, gene, title=f"Spatial Expression of {gene}")














    df_global, df_spot = compare_imputations(
        original_adata,
        [methodA_adata,methodB_adata,methodC_adata,methodD_adata],
        ["Tesla1", "Tesla2", "Tesla3", "Vispro"],
        max_dist=100.0
    )
    print("Global metrics:")
    print(df_global)
    print("Spot-level metrics:")
    print(df_spot)
    # pass

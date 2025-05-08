import anndata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from scipy.spatial import cKDTree
import scanpy as sc
import scipy
from scipy.interpolate import splprep, splev
import pandas as pd
import anndata

import scipy.stats as stats
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind


def get_labels(file):
    adata = sc.read_h5ad(file)
    layer_to_color_map = {'Layer{0}'.format(i + 1): i for i in range(6)}
    layer_to_color_map['WM'] = 6
    labels = list(adata.obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
    original_coor = adata.obsm['spatial']
    coordinates=np.column_stack((original_coor[:,1],original_coor[:,0]))
    # return np.asarray(labels),coordinates
    return adata.obs['layer_guess_reordered'],coordinates



def plot_ARI():
    for imp_file in imputed_files:
        # Load imputed data
        adata_imp = anndata.read_h5ad(imp_file)
        imp_coords = adata_imp.obsm['spatial']
        imp_clusters = adata_imp.obs['kmeans_clusters'].astype(str).values  # ensure categorical handling

        # Propagate clusters quickly using cKDTree
        _, indices = cKDTree(imp_coords).query(gt_coords, k=1)
        propagated_clusters = imp_clusters[indices]

        # Create a consistent color mapping
        unique_clusters = np.unique(np.concatenate([imp_clusters, propagated_clusters]))
        num_colors = len(palette)
        if len(unique_clusters) > num_colors:
            raise ValueError(
                f"Number of clusters ({len(unique_clusters)}) exceeds palette size ({num_colors}). Extend palette.")

        cluster_color_dict = {cluster: palette[i] for i, cluster in enumerate(sorted(unique_clusters))}
        ari = adjusted_rand_score(gt_labels, propagated_clusters)
        print(f"ARI ({imp_file}): {ari:.4f}")

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Left: imputed data clusters
        sns.scatterplot(
            x=imp_coords[:, 1],
            y=imp_coords[:, 0],
            hue=imp_clusters,
            palette=cluster_color_dict,
            linewidth=0,
            s=12,
            ax=ax[0]
        )
        ax[0].set_title("Imputed Spots - KMeans Clusters", fontsize=14, fontweight='bold')
        ax[0].invert_yaxis()
        ax[0].axis('equal')
        ax[0].axis('off')
        ax[0].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Right: propagated clusters at original locations
        sns.scatterplot(
            x=gt_coords[:, 1],
            y=gt_coords[:, 0],
            hue=propagated_clusters,
            palette=cluster_color_dict,
            linewidth=0,
            s=20,
            ax=ax[1]
        )

        ax[1].set_title(f"ARI : {ari:.4f}", fontsize=14, fontweight='bold')
        ax[1].invert_yaxis()
        ax[1].axis('equal')
        ax[1].axis('off')
        ax[1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

# Custom color palette
palette = ['#1b9e77', '#d95f02', '#7570b3',
           '#e7298a', '#66a61e', '#e6ab02', '#a6761d']

# Get ground truth labels and coordinates
gt_labels, gt_coords = get_labels("/media/huifang/data/registration/humanpilot/precessed_feature_matrix/151673.h5ad")

# Imputed dataset files
imputed_files = {
    "Original":"/media/huifang/data/fiducial/tiff_data/151673/knn_cluster/filtered_matrix_with_clusters.h5ad",
    # "/media/huifang/data/fiducial/tiff_data/151673/knn_cluster/filtered_matrix_original_imputed_1_with_clusters.h5ad",
    # "/media/huifang/data/fiducial/tiff_data/151673/knn_cluster/filtered_matrix_original_imputed_2_with_clusters.h5ad",
    # "/media/huifang/data/fiducial/tiff_data/151673/knn_cluster/filtered_matrix_original_imputed_3_with_clusters.h5ad",
    "Vispro": "/media/huifang/data/fiducial/tiff_data/151673/knn_cluster/filtered_matrix_original_imputed_4_with_clusters.h5ad"
}

# Define layer order
layer_order = ['WM', 'Layer6', 'Layer5', 'Layer4', 'Layer3', 'Layer2', 'Layer1']

# Choose MFGE8 as the marker
gene = "MFGE8"
base_palette = ['#1b9e77', '#d95f02']  # Original, Vispro

# Prepare data
dfs = []

for label, path in imputed_files.items():
    adata = anndata.read_h5ad(path)
    imp_coords = adata.obsm['spatial']
    _, indices = cKDTree(gt_coords).query(imp_coords, k=1)
    layer_labels = gt_labels.values[indices]

    gene_expr = adata[:, gene].X
    if not isinstance(gene_expr, np.ndarray):
        gene_expr = gene_expr.toarray()
    gene_expr = gene_expr.flatten()

    df = pd.DataFrame({
        "Layer": layer_labels,
        "MFGE8": gene_expr,
        "Imputation": label
    })
    df["log_MFGE8"] = np.log1p(df["MFGE8"])
    df["Layer"] = pd.Categorical(df["Layer"], categories=layer_order, ordered=True)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Plot
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12,6))
ax = sns.violinplot(
    data=df_all,
    x="Layer",
    y="log_MFGE8",
    hue="Imputation",
    order=layer_order,
    palette=palette,
    cut=0,
    scale="width",
    dodge=True
)

ax.set_title("MFGE8 Expression Across Cortical Layers", fontsize=14)
ax.set_ylabel("log(counts)")
ax.set_xlabel("Layer")
ax.legend(title="Data", bbox_to_anchor=(1.02, 1), fontsize=14, loc='upper left')

# Compute y-range once
y_min = df_all["log_MFGE8"].min()
y_max = df_all["log_MFGE8"].max()
x_positions = np.arange(len(layer_order))

for idx, method in enumerate(["Original", "Vispro"]):
    df_sub = df_all[df_all["Imputation"] == method]

    # Horizontal position offsets
    x_offset = -0.25 if method == "Original" else 0.25

    # Vertical configuration
    if method == "Original":
        y_base = y_max - 0.18  # significantly below violins
        line_height = 0.015   # downward bracket
        text_offset = 0.025   # below the bracket
        va = 'bottom'
    else:
        y_base = y_max + 0.02
        line_height = 0.015
        text_offset = 0.025
        va = 'bottom'

    for i in range(len(layer_order) - 1):
        l1, l2 = layer_order[i], layer_order[i + 1]
        g1 = df_sub[df_sub["Layer"] == l1]["log_MFGE8"]
        g2 = df_sub[df_sub["Layer"] == l2]["log_MFGE8"]

        if len(g1) > 1 and len(g2) > 1:
            _, pval = ttest_ind(g1, g2, equal_var=False)
            if np.isnan(pval):
                continue

            ptxt = f"P = {pval:.2g}" if pval >= 1e-3 else f"P ≤ {pval:.0e}"
            x1, x2 = x_positions[i] + x_offset, x_positions[i + 1] + x_offset
            y = y_base + i * (line_height * 4)  # subtle stagger

            # Draw the bracket
            ax.plot([x1, x1, x2, x2],
                    [y, y + line_height, y + line_height, y],
                    lw=1.1, color=palette[idx])

            # Add the text
            ax.text((x1 + x2) / 2, y + line_height + text_offset, ptxt,
                    ha='center', va=va, fontsize=12, color=palette[idx])
ax.set_ylim(0, 1.8)
plt.tight_layout()
plt.show()

# for imp_file in imputed_files:
#     print(f"Processing: {imp_file}")
#     adata_imp = anndata.read_h5ad(imp_file)
#     imp_coords = adata_imp.obsm['spatial']
#
#     # Propagate layer labels from original (nearest neighbor)
#     _, indices = cKDTree(gt_coords).query(imp_coords, k=1)
#     imp_layer_labels = gt_labels.values[indices]
#
#     # Extract MFGE8 expression
#     mfge8_expr = adata_imp[:, gene].X
#     if not isinstance(mfge8_expr, np.ndarray):
#         mfge8_expr = mfge8_expr.toarray()
#     mfge8_expr = mfge8_expr.flatten()
#
#     # Build DataFrame
#     df = pd.DataFrame({
#         "MFGE8": mfge8_expr,
#         "Layer": imp_layer_labels
#     })
#     df["Layer"] = pd.Categorical(df["Layer"], categories=layer_order, ordered=True)
#     df["log_MFGE8"] = np.log1p(df["MFGE8"])
#
#     # Plot violin
#     plt.figure(figsize=(10, 5))
#     ax = sns.violinplot(x="Layer", y="log_MFGE8", data=df, order=layer_order, palette="Spectral", cut=0)
#     ax.set_ylabel("Log counts")
#     ax.set_title(f"MFGE8 Expression in {imp_file.split('/')[-1]}")
#
#     # Optional: Add pairwise p-values
#     def add_pvals(ax, data, layer_order):
#         ypos = data["log_MFGE8"].max() + 0.2
#         for i in range(len(layer_order) - 1):
#             l1, l2 = layer_order[i], layer_order[i + 1]
#             g1 = data[data["Layer"] == l1]["log_MFGE8"]
#             g2 = data[data["Layer"] == l2]["log_MFGE8"]
#             if len(g1) > 1 and len(g2) > 1:
#                 _, pval = ttest_ind(g1, g2, equal_var=False)
#                 ptxt = f"P = {pval:.2g}" if pval >= 1e-3 else f"P ≤ {pval:.0e}"
#                 x1, x2 = i, i + 1
#                 ax.plot([x1, x1, x2, x2], [ypos, ypos + 0.05, ypos + 0.05, ypos], lw=1.2, color='black')
#                 ax.text((x1 + x2)/2, ypos + 0.08, ptxt, ha='center', va='bottom', fontsize=9)
#                 ypos += 0.25
#
#     add_pvals(ax, df, layer_order)
#
#     plt.tight_layout()
#     plt.show()
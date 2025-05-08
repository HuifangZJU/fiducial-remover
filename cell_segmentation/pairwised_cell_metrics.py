
from sklearn.preprocessing import StandardScaler
import matplotlib.transforms as mtransforms
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib
# matplotlib.use('Agg')  # <- Add this line!
from umap import UMAP
import hdbscan
from hdbscan import approximate_predict

from matplotlib.ticker import AutoMinorLocator

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns

def extract_features(cells):
    features = []

    for cell in cells:
        if np.isnan(cell).any():
            continue

        poly = Polygon(cell)
        if not poly.is_valid or poly.area == 0:
            continue

        area = poly.area
        perimeter = poly.length

        # PCA
        pca = PCA(n_components=2)
        pca.fit(cell)
        eigs = pca.explained_variance_ratio_
        elongation = eigs[0] / (eigs[1] + 1e-8)
        eccentricity = np.sqrt(1 - eigs[1] / (eigs[0] + 1e-8))

        # Convex hull
        try:
            hull = ConvexHull(cell)
            hull_area = Polygon(cell[hull.vertices]).area
        except:
            hull_area = area

        # Bounding box
        minx, miny, maxx, maxy = poly.bounds
        width, height = maxx - minx, maxy - miny
        extent = area / (width * height + 1e-8)
        aspect_ratio = width / (height + 1e-8)

        # Metrics
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
        solidity = area / (hull_area + 1e-8)
        compactness = perimeter**2 / (area + 1e-8)

        features.append([
            area, perimeter, elongation, eccentricity,
            circularity, solidity, extent, aspect_ratio, compactness
        ])

    return np.array(features)

# Project to 2D using PCA
def project_features(features, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        from umap import UMAP
        reducer = UMAP(n_components=2)
    return reducer.fit_transform(features)


def plot_projection(data, title, xlim=None, ylim=None, labels=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    if labels is not None:
        sc = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='Spectral', s=10)
    else:
        sc = ax.scatter(data[:, 0], data[:, 1], s=10)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()
    return fig

def compute_morphology(cell_boundaries):
    sizes = []
    elongations = []
    convexities = []

    for cell in cell_boundaries:
        if np.isnan(cell).any():
            continue  # skip if invalid

        # Create polygon
        polygon = Polygon(cell)
        if not polygon.is_valid or polygon.area == 0:
            continue

        # Size (Area)
        area = polygon.area
        sizes.append(area)

        # Elongation using PCA
        pca = PCA(n_components=2)
        pca.fit(cell)
        elongation = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]
        elongations.append(elongation)

        # Convexity
        try:
            hull = ConvexHull(cell)
            hull_area = Polygon(cell[hull.vertices]).area
            convexity = area / hull_area if hull_area > 0 else 0
        except:
            convexity = 0
        convexities.append(convexity)

    return np.array(sizes), np.array(elongations), np.array(convexities)

def plot_distributions(sizes, elongations, convexities):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(sizes, bins=30)
    axs[0].set_title("Cell Size (Area)")
    axs[1].hist(elongations, bins=30)
    axs[1].set_title("Elongation (Major/Minor Axis Ratio)")
    axs[2].hist(convexities, bins=30)
    axs[2].set_title("Convexity (Area / Convex Hull Area)")
    plt.tight_layout()
    # plt.show()
    return fig


def plot_comparison(proj1, proj2, xlim, ylim, xlabel, ylabel, title_prefix, filename=None):
    """
    Creates a 1×3 subplot:
      [ Original | Vispro | Overlay ]
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # 1) Original
    ax = axes[0]
    ax.scatter(
        proj1[:, 0], proj1[:, 1],
        s=10, edgecolor='k', linewidth=0.2,
        facecolor=base[0], alpha=0.7
    )
    ax.set_title(f"{title_prefix} — Original", pad=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # 2) Vispro
    ax = axes[1]
    ax.scatter(
        proj2[:, 0], proj2[:, 1],
        s=20, edgecolor='k', linewidth=0.2,
        facecolor=base[1], alpha=0.7
    )
    ax.set_title(f"{title_prefix} — Vispro", pad=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # 3) Overlay
    ax = axes[2]
    ax.scatter(
        proj1[:, 0], proj1[:, 1],
        s=20, edgecolor='k', linewidth=0.2,
        facecolor=base[0], alpha=0.5, label='Original'
    )
    ax.scatter(
        proj2[:, 0], proj2[:, 1],
        s=20, edgecolor='k', linewidth=0.2,
        facecolor=base[1], alpha=0.5, label='Vispro'
    )
    ax.set_title(f"{title_prefix} — Overlay", pad=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc='best', fontsize=12)
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if filename:
        fig.savefig(filename, dpi=300)
    plt.show()



def cluster_and_merge_pca_features(locations, pca_features, initial_clusters=5, merge_groups=[1,2,4], figsize=(24, 12)):
    """
    Perform KMeans clustering on PCA features with a 5-cluster setting,
    then merge specified clusters into a single group, and visualize.

    Args:
        locations (np.ndarray): Array of shape (n, 2) for spatial coordinates.
        pca_features (np.ndarray): Array of shape (n, 2) for 2D PCA features.
        initial_clusters (int): Number of initial clusters to form.
        merge_groups (list): List of cluster labels to merge into one group.
        figsize (tuple): Size of the resulting figure.

    Returns:
        None
    """
    kmeans = KMeans(n_clusters=initial_clusters, random_state=42)
    initial_labels = kmeans.fit_predict(pca_features)

    # Step 2: Merge specified clusters
    merged_labels = np.copy(initial_labels)
    target_label = min(merge_groups)
    for idx in range(len(initial_labels)):
        if initial_labels[idx] in merge_groups:
            merged_labels[idx] = target_label

    # Step 3: Relabel to 0, 1, 2
    unique_labels = np.unique(merged_labels)

    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_mapping[label] for label in merged_labels])

    n_final_clusters = len(np.unique(final_labels))
    colors = sns.color_palette("tab10", n_colors=n_final_clusters)

    fig, axs = plt.subplots(2, n_final_clusters + 1, figsize=figsize)

    # Top row: Spatial locations
    for cluster_idx in range(n_final_clusters):
        mask = final_labels == cluster_idx
        axs[0, cluster_idx].scatter(locations[mask, 0], locations[mask, 1],
                                    s=10,  linewidth=0.2, color=colors[cluster_idx], alpha=0.7)
        axs[0, cluster_idx].set_title(f'Spatial Cluster {cluster_idx + 1}')
        axs[0, cluster_idx].axis('off')
        axs[0, cluster_idx].axis('equal')
        axs[0, cluster_idx].invert_yaxis()

    # Overlay spatial
    for cluster_idx in range(n_final_clusters):
        mask = final_labels == cluster_idx
        axs[0, -1].scatter(locations[mask, 0], locations[mask, 1],
                           s=10,  linewidth=0.2,color=colors[cluster_idx], alpha=0.7, label=f'Cluster {cluster_idx + 1}')
    axs[0, -1].set_title('Spatial Overlay')
    # axs[0, -1].legend(fontsize=6)
    axs[0, -1].axis('off')
    axs[0, -1].axis('equal')
    axs[0, -1].invert_yaxis()
    # Bottom row: PCA feature maps
    for cluster_idx in range(n_final_clusters):
        mask = final_labels == cluster_idx
        axs[1, cluster_idx].scatter(pca_features[mask, 0], pca_features[mask, 1],
                                    s=20, edgecolor='k', linewidth=0.2, facecolor=colors[cluster_idx], alpha=0.7)
        axs[1, cluster_idx].set_title(f'PCA Cluster {cluster_idx + 1}')
        axs[1, cluster_idx].axis('off')

    # Overlay PCA
    for cluster_idx in range(n_final_clusters):
        mask = final_labels == cluster_idx
        axs[1, -1].scatter(pca_features[mask, 0], pca_features[mask, 1],
                           s=20, edgecolor='k', linewidth=0.2, facecolor=colors[cluster_idx], alpha=0.7, label=f'Cluster {cluster_idx + 1}')
    axs[1, -1].set_title('PCA Overlay')
    axs[1, -1].legend(fontsize=18)
    # axs[1, -1].axis('off')

    plt.tight_layout()
    extent = axs[1, 3].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    expand = 0.4  # For example, expand by 10% on each side
    new_extent = mtransforms.Bbox.from_extents(
        extent.x0 - expand, extent.y0 - expand,
        extent.x1 + expand, extent.y1 + expand
    )

    fig.savefig('/home/huifang/workspace/code/fiducial_remover/paper_figures/figures/24.png', bbox_inches=new_extent,dpi=600)
    plt.show()

def show_cell_locations(locations):
    colors = sns.color_palette("tab10", n_colors=5)
    # Top row: Spatial locations
    f,a = plt.subplots(figsize=(6,6))
    a.scatter(locations[:, 0], locations[:, 1],
                                    s=10, linewidth=0.2, color=colors[1], alpha=0.7)
    plt.gca().invert_yaxis()
    plt.savefig('/home/huifang/workspace/code/fiducial_remover/paper_figures/figures/25.png',dpi=600)
    plt.show()


def plot_spatial_clusters(locations, pca_features, n_clusters=3, figsize=(8, 6)):
    """
    Cluster 2D PCA features into groups and visualize their spatial locations.

    Args:
        locations (np.ndarray): Array of shape (n, 2) representing spatial coordinates (x, y).
        pca_features (np.ndarray): Array of shape (n, 2) representing 2D PCA features.
        n_clusters (int): Number of clusters to form (default: 3).
        figsize (tuple): Size of the plot figure (default: (8,6)).

    Returns:
        None
    """

    # Perform k-means clustering
    # gmm = GaussianMixture(n_components=n_clusters, random_state=12)
    # labels = gmm.fit_predict(pca_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pca_features)

    # Define distinct colors
    colors = sns.color_palette("tab10", n_colors=n_clusters)

    # Create subplots: 2 rows (spatial and PCA) x (n_clusters + 1) columns
    fig, axs = plt.subplots(2, n_clusters + 1, figsize=figsize)

    # --------- Plot spatial clusters ----------
    for cluster_idx in range(n_clusters):
        mask = labels == cluster_idx
        axs[0, cluster_idx].scatter(locations[mask, 0], locations[mask, 1],
                                    color=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}', s=10)
        axs[0, cluster_idx].set_title(f'Spatial Cluster {cluster_idx + 1}')
        axs[0, cluster_idx].axis('off')

    # Spatial overlay
    for cluster_idx in range(n_clusters):
        mask = labels == cluster_idx
        axs[0, -1].scatter(locations[mask, 0], locations[mask, 1],
                           color=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}', s=10)
    axs[0, -1].set_title('Spatial Overlay')
    axs[0, -1].legend(fontsize=8)
    axs[0, -1].axis('off')

    # --------- Plot PCA clusters ----------
    for cluster_idx in range(n_clusters):
        mask = labels == cluster_idx
        axs[1, cluster_idx].scatter(pca_features[mask, 0], pca_features[mask, 1],
                                    color=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}', s=10)
        axs[1, cluster_idx].set_title(f'PCA Cluster {cluster_idx + 1}')
        axs[1, cluster_idx].axis('off')

    # PCA overlay
    for cluster_idx in range(n_clusters):
        mask = labels == cluster_idx
        axs[1, -1].scatter(pca_features[mask, 0], pca_features[mask, 1],
                           color=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}', s=10)
    axs[1, -1].set_title('PCA Overlay')
    axs[1, -1].legend(fontsize=8)
    axs[1, -1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_spatial_and_pca_dbscan_paired(locations1, pca_1,locations2,pca_2, eps=0.8, min_samples=10, figsize=(14, 14)):
    """
    Cluster 2D PCA features into groups using DBSCAN and visualize both their spatial locations and PCA feature projections.

    Args:
        locations (np.ndarray): Array of shape (n, 2) representing spatial coordinates (x, y).
        pca_features (np.ndarray): Array of shape (n, 2) representing 2D PCA features.
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): Number of samples in a neighborhood for a point to be considered a core point.
        figsize (tuple): Size of the overall figure.

    Returns:
        None
    """
    pca_features = np.concatenate((pca_1, pca_2), axis=0)
    pca_features = pca_features[:,:8]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=1, prediction_data=True)
    clusterer.fit(pca_features)
    labels = approximate_predict(clusterer, pca_features)[0]
    # labels = labels+1

    unique_labels = np.unique(labels)

    n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise

    # Define distinct colors
    colors = sns.color_palette("tab10", n_colors=max(n_clusters, 1))

    n1 = pca_1.shape[0]
    labels1,labels2 = labels[:n1], labels[n1:]


    # Create subplots: 2 rows (spatial and PCA) x (clusters + overlay)
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    for cluster_idx in unique_labels:
        mask1 = labels1 == cluster_idx
        if cluster_idx == -1:
            color = 'lightgray'
            label_name = 'Noise'
        else:
            color = colors[cluster_idx % len(colors)]
            label_name = f'Cluster {cluster_idx}'
        axs[0, 0].scatter(locations1[mask1, 0], locations1[mask1, 1],
                           color=color, label=label_name, s=10)
    axs[0, 0].set_title('Spatial Overlay of Original Data')
    axs[0, 0].legend(fontsize=6)
    axs[0, 0].axis('off')
    axs[0,0].invert_yaxis()

    # PCA overlay including noise
    for cluster_idx in unique_labels:
        mask1 = labels1 == cluster_idx
        if cluster_idx == -1:
            color = 'lightgray'
            label_name = 'Noise'
        else:
            color = colors[cluster_idx % len(colors)]
            label_name = f'Cluster {cluster_idx}'
        axs[0, -1].scatter(pca_1[mask1, 0], pca_1[mask1, 1],
                           color=color, label=label_name, s=10)
    axs[0, -1].set_title('PCA Overlay')
    axs[0, -1].legend(fontsize=6)
    axs[0, -1].axis('off')


    # for the second type of data

    for cluster_idx in unique_labels:
        mask2 = labels2 == cluster_idx
        if cluster_idx == -1:
            color = 'lightgray'
            label_name = 'Noise'
        else:
            color = colors[cluster_idx % len(colors)]
            label_name = f'Cluster {cluster_idx}'
        axs[1, 0].scatter(locations2[mask2, 0], locations2[mask2, 1],
                          color=color, label=label_name, s=10)
    axs[1, 0].set_title('Spatial Overlay of Vispro Data')
    axs[1, 0].legend(fontsize=6)
    axs[1, 0].axis('off')
    axs[1, 0].invert_yaxis()
    # PCA overlay including noise
    for cluster_idx in unique_labels:
        mask2 = labels2 == cluster_idx
        if cluster_idx == -1:
            color = 'lightgray'
            label_name = 'Noise'
        else:
            color = colors[cluster_idx % len(colors)]
            label_name = f'Cluster {cluster_idx}'
        axs[1, -1].scatter(pca_2[mask2, 0], pca_2[mask2, 1],
                           color=color, label=label_name, s=10)
    axs[1, -1].set_title('PCA Overlay')
    axs[1, -1].legend(fontsize=6)
    axs[1, -1].axis('off')

    plt.tight_layout()
    plt.show()



def plot_spatial_and_pca_dbscan(locations, pca_features, eps=0.8, min_samples=10, figsize=(18, 8)):
    """
    Cluster 2D PCA features into groups using DBSCAN and visualize both their spatial locations and PCA feature projections.

    Args:
        locations (np.ndarray): Array of shape (n, 2) representing spatial coordinates (x, y).
        pca_features (np.ndarray): Array of shape (n, 2) representing 2D PCA features.
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): Number of samples in a neighborhood for a point to be considered a core point.
        figsize (tuple): Size of the overall figure.

    Returns:
        None
    """
    # Perform DBSCAN clustering
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # labels = dbscan.fit_predict(pca_features)

    # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    # labels = clusterer.fit_predict(pca_features)


    # pca_features = pca_features[:,:3]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=1, prediction_data=True)
    clusterer.fit(pca_features)
    labels = approximate_predict(clusterer, pca_features)[0]

    unique_labels = np.unique(labels)

    n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise

    # Define distinct colors
    colors = sns.color_palette("tab10", n_colors=max(n_clusters, 1))

    # Create subplots: 2 rows (spatial and PCA) x (clusters + overlay)
    fig, axs = plt.subplots(2, n_clusters + 1, figsize=figsize)

    # Plot spatial clusters individually
    cluster_indices = unique_labels[unique_labels != -1]  # Exclude noise for individual plots
    # cluster_indices = unique_labels
    for idx, cluster_idx in enumerate(cluster_indices):
        mask = labels == cluster_idx
        axs[0, idx].scatter(locations[mask, 0], locations[mask, 1],
                            color=colors[idx % len(colors)], label=f'Cluster {cluster_idx}', s=10)
        axs[0, idx].set_title(f'Spatial Cluster {cluster_idx}')
        axs[0, idx].axis('off')

    # Spatial overlay including noise
    for cluster_idx in unique_labels:
        mask = labels == cluster_idx
        if cluster_idx == -1:
            color = 'lightgray'
            label_name = 'Noise'
        else:
            color = colors[cluster_idx % len(colors)]
            label_name = f'Cluster {cluster_idx}'
        axs[0, -1].scatter(locations[mask, 0], locations[mask, 1],
                           color=color, label=label_name, s=10)
    axs[0, -1].set_title('Spatial Overlay')
    axs[0, -1].legend(fontsize=6)
    axs[0, -1].axis('off')

    # Plot PCA clusters individually
    for idx, cluster_idx in enumerate(cluster_indices):
        mask = labels == cluster_idx
        axs[1, idx].scatter(pca_features[mask, 0], pca_features[mask, 1],
                            color=colors[idx % len(colors)], label=f'Cluster {cluster_idx}', s=10)
        axs[1, idx].set_title(f'PCA Cluster {cluster_idx}')
        axs[1, idx].axis('off')

    # PCA overlay including noise
    for cluster_idx in unique_labels:
        mask = labels == cluster_idx
        if cluster_idx == -1:
            color = 'lightgray'
            label_name = 'Noise'
        else:
            color = colors[cluster_idx % len(colors)]
            label_name = f'Cluster {cluster_idx}'
        axs[1, -1].scatter(pca_features[mask, 0], pca_features[mask, 1],
                           color=color, label=label_name, s=10)
    axs[1, -1].set_title('PCA Overlay')
    axs[1, -1].legend(fontsize=6)
    axs[1, -1].axis('off')

    plt.tight_layout()
    plt.show()

def get_feature_pca_umap(cells_original, cells_vispro2):
    features_original = extract_features(cells_original)
    features_vispro2 = extract_features(cells_vispro2)
    features_all = np.vstack([features_original, features_vispro2])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_all)
    # PCA (shared)
    pca_proj = PCA(n_components=9).fit_transform(features_scaled)
    # UMAP (shared)
    umap_proj = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(features_scaled)

    n1 = features_original.shape[0]
    pca_1, pca_2 = pca_proj[:n1], pca_proj[n1:]
    umap_1, umap_2 = umap_proj[:n1], umap_proj[n1:]
    return pca_1,pca_2,umap_1,umap_2

def readfromfile(path):
    data = np.load(path)
    pca_1 = data["pca_1"]
    pca_2 = data["pca_2"]
    umap_1 = data["umap_1"]
    umap_2 = data["umap_2"]
    return pca_1,pca_2,umap_1,umap_2

def get_data(path):
    data = np.load(path)
    cell_boundaries = data['boundary']
    cell_centers = data['center']
    return cell_centers,cell_boundaries
# Example usage
path = "/media/huifang/data/fiducial/temp_result/vispro/cell_segmentation/"
# for i in range(8,20):
base = ['#1f77b4', '#ff7f0e']

# Apply a clean, scientific plotting style
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
})

for i in [4]:
    print(i)


    cell_locations_original, cell_boundary_original = get_data(path + "original/" + str(i) + '_original.npz')
    cell_locations_vispro, cell_boundary_vispro = get_data(path + "vispro2/" + str(i) + '_vispro2.npz')
    # show_cell_locations(cell_locations_vispro)
    # pca_1, pca_2 ,umap_1, umap_2 = get_feature_pca_umap(cell_boundary_original,cell_boundary_vispro)
    # np.savez_compressed(
    #     path + str(i) + "projections.npz",
    #     pca_1=pca_1,
    #     pca_2=pca_2,
    #     umap_1=umap_1,
    #     umap_2=umap_2
    # )
    # print('saved')
    # test = input()


    pca_1, pca_2, umap_1, umap_2 = readfromfile(path+str(i)+"projections.npz")
    cluster_and_merge_pca_features(cell_locations_original, pca_1[:,:2])


    # plot_spatial_and_pca_dbscan(cell_locations_vispro, pca_2)
    # plot_spatial_and_pca_dbscan(cell_locations_original, pca_1)

    plot_spatial_and_pca_dbscan_paired(cell_locations_original, pca_1, cell_locations_vispro,pca_2)

    pca = np.concatenate((pca_1,pca_2 ), axis=0)
    position=np.concatenate((cell_locations_original,cell_locations_vispro ), axis=0)
    plot_spatial_and_pca_dbscan(position, pca)
    # plot_spatial_and_pca_dbscan(cell_locations_vispro, pca_2)
    plot_spatial_and_pca_dbscan(cell_locations_original, pca_1)

    # show_cell_locations(cell_locations_vispro)
    plot_spatial_clusters(cell_locations_original,pca_1,n_clusters=3)

    # plot_spatial_and_pca_dbscan(cell_locations_original,pca_1)
    # cluster_and_merge_pca_features(cell_locations_original,pca_1)

    print(cell_locations_original.shape)
    print(cell_locations_vispro.shape)
    print(pca_1.shape)
    print(pca_2.shape)
    test = input()




    # Calculate shared limits
    x_min = min(pca_1[:, 0].min(), pca_2[:, 0].min())
    x_max = max(pca_1[:, 0].max(), pca_2[:, 0].max())
    y_min = min(pca_1[:, 1].min(), pca_2[:, 1].min())
    y_max = max(pca_1[:, 1].max(), pca_2[:, 1].max())
    pca_xlim, pca_ylim = (x_min-2, x_max+2), (y_min-2, y_max+2)

    x_min = min(umap_1[:, 0].min(), umap_2[:, 0].min())
    x_max = max(umap_1[:, 0].max(), umap_2[:, 0].max())
    y_min = min(umap_1[:, 1].min(), umap_2[:, 1].min())
    y_max = max(umap_1[:, 1].max(), umap_2[:, 1].max())
    umap_xlim, umap_ylim = (x_min-2, x_max+2), (y_min-2, y_max+2)

    # fig = plot_projection(pca_1, title='PCA - Original', xlim=pca_xlim, ylim=pca_ylim)
    # fig.savefig(os.path.join(path, f'{i}_pca_original.png'))
    # plt.close()
    #
    # fig = plot_projection(pca_2, title='PCA - Vispro', xlim=pca_xlim, ylim=pca_ylim)
    # fig.savefig(os.path.join(path, f'{i}_pca_vispro2.png'))
    # plt.close()
    #
    # fig = plot_projection(umap_1, title='UMAP - Original', xlim=umap_xlim, ylim=umap_ylim)
    # fig.savefig(os.path.join(path, f'{i}_umap_original.png'))
    # plt.close()
    #
    # fig = plot_projection(umap_2, title='UMAP - Vispro', xlim=umap_xlim, ylim=umap_ylim)
    # fig.savefig(os.path.join(path, f'{i}_umap_vispro2.png'))
    # plt.close()

    # Usage examples:
    plot_comparison(
        pca_1, pca_2,
        pca_xlim, pca_ylim,
        "PC 1", "PC 2",
        "PCA",
        filename="pca_comparison.png"
    )

    plot_comparison(
        umap_1, umap_2,
        umap_xlim, umap_ylim,
        "UMAP 1", "UMAP 2",
        "UMAP",
        filename="umap_comparison.png"
    )



    # combined_labels = ['set1'] * n1 + ['set2'] * (features_all.shape[0] - n1)
    # plot_projection(pca_proj, labels=combined_labels, title='Combined PCA View')
    # fig.savefig(os.path.join(path, f'{i}_comparison.png'))
    # plt.close()
    # test = input()

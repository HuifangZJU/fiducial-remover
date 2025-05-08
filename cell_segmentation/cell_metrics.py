
from sklearn.preprocessing import StandardScaler
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # <- Add this line!
import matplotlib.pyplot as plt
from umap import UMAP
import hdbscan
import os

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

# Visualize
def plot_projection(projected, labels=None, title="Morphology Projection"):
    fig =plt.figure(figsize=(6, 5))
    if labels is not None:
        plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap='viridis', s=10)
    else:
        plt.scatter(projected[:, 0], projected[:, 1], s=10)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    # plt.show()
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
# Example usage
path = "/media/huifang/data/fiducial/temp_result/vispro/cell_segmentation/"
# for i in range(8,20):
for i in [12]:
    print(i)
    # if i==8:
    #     continue
    # data = np.load(path +'sample_data.npz')
    # data = np.load(path + str(i)+ '_original.npz')
    data = np.load(path + str(i) + '_vispro2.npz')
    centers = data['center']
    cell_boundaries = data['boundary']


    # # Morphology metrics
    # sizes, elongations, convexities = compute_morphology(cell_boundaries)
    # fig = plot_distributions(sizes, elongations, convexities)
    # fig.savefig(os.path.join(path, f'{i}_morpho_distributions_vispro2.png'))
    # plt.close(fig)


    # Basic PCA projection
    features = extract_features(cell_boundaries)
    projected = project_features(features, method='pca')
    fig =plot_projection(projected)
    fig.savefig(os.path.join(path, f'{i}_pca_projection_basic_vispro2.png'))
    plt.close()

    # Enhanced PCA
    features_scaled = StandardScaler().fit_transform(features)
    pca_proj = PCA(n_components=2).fit_transform(features_scaled)
    fig =plot_projection(pca_proj, title="Enhanced PCA of Cell Morphology")
    fig.savefig(os.path.join(path, f'{i}_pca_projection_enhanced_vispro2.png'))
    plt.close()

    # UMAP projection
    umap_proj = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(features_scaled)
    fig =plot_projection(umap_proj, title="UMAP of Cell Morphologies")
    fig.savefig(os.path.join(path, f'{i}_umap_projection_vispro2.png'))
    plt.close()
    #
    # # UMAP + HDBSCAN Clustering
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    # labels = clusterer.fit_predict(features_scaled)
    # fig =plot_projection(umap_proj, labels=labels, title="UMAP + HDBSCAN Clusters")
    # fig.savefig(os.path.join(path, f'{i}_umap_hdbscan_clusters.png'))
    # plt.close()

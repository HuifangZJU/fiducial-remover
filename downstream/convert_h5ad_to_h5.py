import pandas as pd
import h5py
from anndata import read_h5ad


def save_as_h5(adata_file_name):
    # Load the .h5ad file
    adata = read_h5ad(adata_file_name)

    # Create a new HDF5 file in 'write' mode
    with h5py.File(adata_file_name.replace(".h5ad", "_reverted.h5"), "w") as h5file:
        # Save the main data matrix (X)
        h5file.create_dataset("X", data=adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

        # Save genes and barcodes as datasets (or any necessary observations and variables)
        h5file.create_dataset("gene_names", data=[name.encode() for name in adata.var_names])
        h5file.create_dataset("barcodes", data=[name.encode() for name in adata.obs_names])

        # Save spatial coordinates if they exist in adata
        if "pixel_x" in adata.obs and "pixel_y" in adata.obs:
            spatial_data = pd.DataFrame({
                "in_tissue": adata.obs["x1"],
                "array_row": adata.obs["array_x"],
                "array_col": adata.obs["array_y"],
                "pxl_row_in_fullres": adata.obs["pixel_x"],
                "pxl_col_in_fullres": adata.obs["pixel_y"]
            })
            spatial_file_name = adata_file_name.replace(".h5ad", "_spatial.csv")
            spatial_data.to_csv(spatial_file_name, sep=",", header=False, index=False)

        # Optionally save other metadata, such as gene names or additional observations
        h5file.create_dataset("gene_ids", data=[name.encode() for name in adata.var["genename"]])

    print(
        f"Data saved as {adata_file_name.replace('.h5ad', '_reverted.h5')} and spatial coordinates as {spatial_file_name}")

adata_file = '/media/huifang/data/fiducial/tiff_data/151672/filtered_matrix_original_imputed_4.h5ad'
save_as_h5(adata_file)
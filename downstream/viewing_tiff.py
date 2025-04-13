import numpy as np
import tifffile
import cv2
from matplotlib import pyplot as plt

tiff_path = "/media/huifang/data/fiducial/tiff_data/151673/151673_full_image_recovered.tif"
with tifffile.TiffFile(tiff_path) as tif:
    img_data = tif.asarray()  # Numpy array of the image
    ome_metadata = tif.ome_metadata  # OME-XML (unused in this example)

print("Original image shape:", img_data.shape)

img_data_cv = img_data.astype(np.uint8)
# Desired new size
new_width = img_data.shape[1] // 25
new_height = img_data.shape[0] // 25

# Resize using INTER_AREA (good for downsampling)
resized_img = cv2.resize(img_data_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)

print("Resized image shape:", resized_img.shape)

plt.imshow(resized_img)
plt.show()


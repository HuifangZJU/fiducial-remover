import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.color import label2rgb
from skimage.transform import resize
from PIL import Image

def get_rgb_img(path):
    rgba_image = Image.open(path)

    # Create a new background image (white) with the same size as the RGBA image
    background = Image.new('RGBA', rgba_image.size, (255, 255, 255, 255))  # RGBA with white background

    # Blend the RGBA image with the white background
    blended_image = Image.alpha_composite(background, rgba_image).convert('RGB')

    # Convert the blended RGB image to a NumPy array
    rgb_array = np.array(blended_image)
    return rgb_array

# Display the NumPy array




# Example RGB image
rgb_image = plt.imread('/home/huifang/workspace/code/fiducial_remover/location_annotation/17.png')
# rgb_image = get_rgb_img('/home/huifang/workspace/code/backgroundremover/bgrm_out/17.png')
# Example binary mask
img = plt.imread('/home/huifang/workspace/code/fiducial_remover/temp_result/application/bgrm-backup/17.png')

binary_mask = img[:, :, 3]
binary_mask[img[:, :, 3]> 0.3] = 1
binary_mask[img[:, :, 3]< 0.3]=0
binary_mask = resize(binary_mask, rgb_image.shape[:2], order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

# Find connected components
labeled_mask, num_features = label(binary_mask)

# Print number of disconnected components
print(f"Number of disconnected components: {num_features}")

# Overlay labeled mask on RGB image
overlay = label2rgb(labeled_mask, image=rgb_image, bg_label=0, alpha=0.5, kind='overlay')

# Plot the results
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(rgb_image)
axes[0].set_title('Original Image')
axes[0].axis('off')
# Original binary mask
axes[1].imshow(binary_mask, cmap='gray')
axes[1].set_title('Binary Mask')
axes[1].axis('off')

# Labeled mask
axes[2].imshow(labeled_mask, cmap='nipy_spectral')
axes[2].set_title('Labeled Mask')
axes[2].axis('off')

# Overlay on RGB image
axes[3].imshow(overlay)
axes[3].set_title('Overlay on RGB Image')
axes[3].axis('off')

plt.show()

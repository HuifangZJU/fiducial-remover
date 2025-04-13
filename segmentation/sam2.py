import cv2
import numpy as np
import matplotlib.pyplot as plt

# Imports from Segment Anything
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ----------------------------
# 1. Load the SAM model
# ----------------------------
checkpoint_path = "/media/huifang/data/fiducial/sam_model/sam_vit_h_4b8939.pth"  # path to your downloaded checkpoint
model_type = "vit_h"                      # choose "vit_h", "vit_l", or "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device="cuda")  # or "cpu" if you don't have a GPU

mask_generator = SamAutomaticMaskGenerator(sam)

# mask_generator = SamAutomaticMaskGenerator(
#     sam,
#     points_per_side=16,           # fewer sampling points
#     pred_iou_thresh=0.50,         # stricter IoU threshold for predicted masks
#     stability_score_thresh=0.97,  # stricter stability requirement
#     min_mask_region_area=10000    # ignore small regions (area in pixels)
# )


# ----------------------------
# 2. Read your image
# ----------------------------
for i in range(167):
    print(i)
    # original_image = cv2.imread("/media/huifang/data/fiducial/annotations/location_annotation/"+str(i)+".png")  # BGR format
    original_image = cv2.imread("/media/huifang/data/fiducial/tiff_data/151508/spatial/tissue_hires_image_0.png")  # BGR format by default
    # Suppose you want to downscale to 768x768
    desired_width = 768
    desired_height = 768

    # 2. Downsize (maintain aspect ratio if you wish)
    # This snippet just forces a fixed 768x768 output
    resized_image = cv2.resize(
        original_image,
        (desired_width, desired_height),
        interpolation=cv2.INTER_AREA  # good for downscaling
    )
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # ----------------------------
    # 3. Automatically generate masks
    # ----------------------------
    masks = mask_generator.generate(image_rgb)
    # 'masks' is a list of dicts, each containing:
    #   - 'segmentation': a 2D boolean mask for one object
    #   - 'bbox': [x, y, w, h] bounding box
    #   - 'area': area of the mask
    #   - 'predicted_iou', 'stability_score', etc.

    print(f"Found {len(masks)} masks in the image.")

    # ----------------------------
    # 4. Visualize the results
    # ----------------------------
    # Let's create a simple color overlay for all masks together.

    # We'll build an index map where each pixel is 0 if no mask, or i if it belongs to masks[i].
    # If a pixel belongs to multiple masks, the last one in the loop wins (simple override).
    height, width = image_rgb.shape[:2]
    index_map = np.zeros((height, width), dtype=np.uint32)

    for i, mask_data in enumerate(masks, start=1):
        segmentation = mask_data["segmentation"]
        index_map[segmentation] = i

    # Generate random colors (R, G, B) for each mask, plus black for background index 0
    random_colors = np.random.randint(0, 255, size=(len(masks)+1, 3), dtype=np.uint8)
    # random_colors[0] = [0, 0, 0]  # optional: make index 0 black

    # Convert the index_map to an RGB image using our random color palette
    colored_result = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            colored_result[i, j] = random_colors[index_map[i, j]]

    # Show the original and the color-labeled masks with matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.imshow(colored_result, alpha=0.5)  # overlay with transparency
    plt.title(f"All Masks (Total: {len(masks)})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

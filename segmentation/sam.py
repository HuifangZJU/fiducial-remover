import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
# Imports from Segment Anything
from segment_anything import sam_model_registry, SamPredictor

# ----------------------------
# 1. Load the model checkpoint
# ----------------------------

checkpoint_path = "/media/huifang/data/fiducial/sam_model/sam_vit_h_4b8939.pth"  # Path to your downloaded checkpoint
model_type = "vit_h"                      # Model type: "vit_h", "vit_l", or "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device="cuda")  # or "cpu" if you don't have a GPU

# ----------------------------
# 2. Load an image
# ----------------------------
for i in range(167):
    print(i)
    # image = cv2.imread("/home/huifang/workspace/code/fiducial_remover/temp_result/application/model_out/recovery/"+str(i)+".png")  # BGR format
    image = cv2.imread("/media/huifang/data/fiducial/tiff_data/151508/spatial/tissue_hires_image_0.png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR -> RGB

    # ----------------------------
    # 3. Create a predictor
    # ----------------------------

    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # ----------------------------
    # 4. Provide a prompt (e.g. a single point)
    # ----------------------------

    # For example, let's say you know there's an object of interest around pixel (x=100, y=200).
    height, width, _ = image_rgb.shape
    # The center coordinates (x, y)
    center_x = width // 2
    center_y = height // 2
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])  # 1 indicates a foreground (positive) point

    # ----------------------------
    # 5. Make a prediction
    # ----------------------------

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True  # If True, returns multiple mask candidates
    )

    # masks is a list of binary mask arrays (shape: [num_masks, height, width])
    # scores is the model's confidence scores for each mask
    # logits is the raw predictions

    # ----------------------------
    # 6. Visualize or save results
    # ----------------------------
    # 1. Identify the highest-scoring mask
    print(len(scores))
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    # 2. Plot: show the image and overlay the best mask
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.imshow(best_mask, alpha=0.5)  # overlay the mask with partial transparency
    plt.title(f"Best Mask (Index={best_idx}), Score={best_score:.3f}")
    plt.axis("off")
    plt.show()
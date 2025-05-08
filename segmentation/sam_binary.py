import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

def get_sam_mask(img_path,sam):
    image = cv2.imread(img_path)
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
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True  # If True, returns multiple mask candidates
    )
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    mask_255 = (best_mask.astype(np.uint8)) * 255
    return mask_255

checkpoint_path = "/media/huifang/data/fiducial/sam_model/sam_vit_h_4b8939.pth"  # Path to your downloaded checkpoint
model_type = "vit_h"                      # Model type: "vit_h", "vit_l", or "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device="cuda")  # or "cpu" if you don't have a GPU


test_image_path= '/home/huifang/workspace/data/imagelists/vispro/tissue_segmentation.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
saving_path = "/media/huifang/data/fiducial/temp_result/vispro/segmentation/"

for i in range(len(files)):
    print(i)
    line = files[i].rstrip().split(' ')
    sample_id = int(line[0])
    # image_path = line[1]
    image_path = "/media/huifang/data/fiducial/temp_result/application/model_out/recovery/" + str(sample_id) + '.png'
    mask_sam = get_sam_mask(image_path,sam)
    cv2.imwrite(saving_path+str(sample_id)+'_sam_binary.png', mask_sam)
    # plt.imshow(mask_otsu)
    # plt.show()

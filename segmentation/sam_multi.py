import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def get_sam_mask(img_path,sam):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    # 2. Compute a scale factor so the longer side is at most 768
    scale = min(768 / orig_h, 768 / orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    # 3. Resize down
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. Generate masks on the smaller image
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(resized)

    # 5. Build an index map at the original size
    index_map = np.zeros((orig_h, orig_w), dtype=np.uint8)

    for i, mask_data in enumerate(masks, start=1):
        small_seg = mask_data["segmentation"].astype(np.uint8)  # H’×W’
        # 6. Upsample mask back to original with nearest‐neighbor
        full_seg = cv2.resize(
            small_seg,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        index_map[full_seg] = i
    return index_map


checkpoint_path = "/media/huifang/data/fiducial/sam_model/sam_vit_h_4b8939.pth"  # path to your downloaded checkpoint
model_type = "vit_h"                      # choose "vit_h", "vit_l", or "vit_b"
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
    cv2.imwrite(saving_path+str(sample_id)+'_sam_multi.png', mask_sam)
    # plt.imshow(mask_otsu)
    # plt.show()

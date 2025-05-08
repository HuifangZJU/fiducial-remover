import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_otsu_mask(img_path):
    # Convert to grayscale for thresholding
    original_bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    # 2) Apply a blur to reduce noise (optional but recommended)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3) Otsu's thresholding to get a binary mask
    _, binary_mask = cv2.threshold(
        gray_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return 255-binary_mask


test_image_path= '/home/huifang/workspace/data/imagelists/vispro/tissue_segmentation.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
saving_path = "/media/huifang/data/fiducial/temp_result/vispro/segmentation/"

for i in range(len(files)):

    line = files[i].rstrip().split(' ')
    sample_id = int(line[0])
    # image_path = line[1]
    image_path = "/media/huifang/data/fiducial/temp_result/application/model_out/recovery/"+str(sample_id)+'.png'
    mask_otsu = get_otsu_mask(image_path)
    cv2.imwrite(saving_path+str(sample_id)+'_otsu.png', mask_otsu)
    # plt.imshow(mask_otsu)
    # plt.show()

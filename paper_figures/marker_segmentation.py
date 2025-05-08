import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
def calculate_normalized_iou(mask1, mask2):

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 1.0  # Avoid division by zero

    return iou

def get_overlayed_image(mask,output_dir,color):
    img = plt.imread(image_name.split(' ')[0])

    target_height =img.shape[0]
    target_width = img.shape[1]

    # Resize binary mask
    mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


    # If image is float (range 0–1), convert to uint8
    if img.dtype != np.uint8:
        img = (img-img.min())/(img.max()-img.min())
        img = (img * 255).astype(np.uint8)

    # Ensure it's a 3-channel image
    if img.ndim == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # has alpha channel
        img = img[:, :, :3]

    mask_bool = mask.astype(bool)

    green_mask = np.zeros_like(img, dtype=np.uint8)
    green_mask[mask_bool] = color  # RGB
    # green_mask[mask_bool] = [250, 0, 0]  # RGB

    alpha = 0.5
    overlay_image = cv2.addWeighted(green_mask, alpha, img, 1-alpha, 0)

    overlay_image = overlay_image[-250:,:250,:]
    # overlay_image = overlay_image[0:250, 20:270, :]

    cv2.imwrite(output_dir, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
    #
    # plt.imshow(overlay_image)
    # plt.show()


def binarize_mask(mask_array, threshold=0.5):

    binary_mask = (mask_array > threshold).astype(np.uint8)
    return binary_mask

test_image_path = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
num_files = len(files)

cellpose_folder = "/media/huifang/data/fiducial/cellpose_results/masks/"
circlenet_folder = "/media/huifang/data/fiducial/circle_net_results/"

save_path="/media/huifang/data/fiducial/temp_result/vispro/marker_segmentation/"
iou_houghs=[]
iou_cellposes=[]
iou_circlenets=[]
for i in range(24,num_files):
    print(i)
    start_time = time.time()
    image_name = files[i]

    label_percentage = float(image_name.split(' ')[1])

    if not os.path.exists(image_name.split(' ')[0].split('.')[0] + '_10x.png'):
        continue

    ground_truth =  binarize_mask(plt.imread(image_name.split(' ')[0].split('.')[0] + '_ground_truth.png'))

    vispro_mask = binarize_mask(plt.imread('/media/huifang/data/fiducial/vispro_masks/'+str(i)+'_0.png'))

    mask_10x = binarize_mask(plt.imread(image_name.split(' ')[0].split('.')[0] + '_10x.png'))
    mask_hough = binarize_mask(plt.imread(image_name.split(' ')[0].split('.')[0] + '_auto_tight.png'))
    mask_cellpose = binarize_mask(plt.imread(cellpose_folder+str(i)+"_cp_masks.png"))
    mask_circleNet = binarize_mask(plt.imread(circlenet_folder+str(i)+".png"))
    # #
    get_overlayed_image(mask_10x, save_path + str(i) + '_10x_croped2.png',[0,200,200])
    get_overlayed_image(ground_truth, save_path + str(i) + '_gt_croped2.png',[255,0,0])
    get_overlayed_image(mask_hough,save_path+str(i)+'_hough_croped2.png',[0,200,200])
    get_overlayed_image(mask_cellpose, save_path + str(i) + '_cellpose_croped2.png',[0,200,200])
    get_overlayed_image(mask_circleNet, save_path + str(i) + '_circleNet_croped2.png',[0,200,200])
    get_overlayed_image(vispro_mask,save_path+str(i)+'_vispro_croped2.png',[0,200,200])
    print('saved')
    test = input()

    # f,a=plt.subplots(1,3)
    # a[0].imshow(mask_hough)
    # a[1].imshow(mask_cellpose)
    # a[2].imshow(mask_circleNet)
    # plt.show()
    # continue



    # iou_cellpose = calculate_normalized_iou(ground_truth,mask_cellpose)
    # iou_hough = calculate_normalized_iou(ground_truth,mask_hough)
    # iou_circlenet = calculate_normalized_iou(ground_truth,mask_circleNet)
    #
    # iou_houghs.append(iou_hough)
    # iou_circlenets.append(iou_circlenet)
    # iou_cellposes.append(iou_cellpose)

#
# np.save('./result/iou_houghs.npy',np.asarray(iou_houghs))
# np.save('./result/iou_circlenets.npy',np.asarray(iou_circlenets))
# np.save('./result/iou_cellposes.npy',np.asarray(iou_cellposes))


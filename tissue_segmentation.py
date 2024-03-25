import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from scipy import ndimage
from skimage import io, color, filters
# Load an H&E stained image
def read_labelme_json(json_file, image_shape, scale,label='tissue'):
    with open(json_file) as file:
        data = json.load(file)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Assuming grayscale mask
    for shape in data['shapes']:
        if shape['label'] == label:
            polygon = np.array(shape['points'])
            polygon[: ,0] = polygon[:, 0]*scale[0]
            polygon[:, 1] = polygon[:, 1] * scale[1]
            polygon = np.asarray(polygon,dtype=np.int32)
            cv2.fillPoly(mask, [polygon], color=255)
    mask = (mask > 128)
    return mask

def find_nearest_multiple_of_32(x):
    base = 32
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score



def get_mask(path):
    img_pil = Image.open(path)
    h, w = img_pil.size
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new), Image.ANTIALIAS)
    mask = np.array(img_pil)
    # mask = plt.imread(path)
    mask = mask[:,:,3]
    threshold_value = 140  # Adjust the threshold value as needed
    mask = (mask> threshold_value)

    return mask

def get_image(path,return_ratio=False):
    img_pil = Image.open(path)
    h, w = img_pil.size
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new), Image.ANTIALIAS)
    image = np.array(img_pil)
    # Normalize the image if necessary
    image = image / 255.0 if np.max(image) > 1 else image
    if return_ratio:
        return image,h_new/h,w_new/w
    else:
        return image

def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice Coefficient.
    :param y_true: Ground truth (binary).
    :param y_pred: Predictions (binary).
    :return: Dice coefficient.
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def get_overlayed_image(image,mask):
    # Create a color mask
    colored_mask = np.zeros_like(image)  # an array of zeros with the same shape as the image
    colored_mask[mask == 1] = [0, 0, 1]  # for example, red color for the mask

    # Overlay the mask on the image
    alpha = 0.4 # transparency level
    overlayed_image = (1 - alpha) * image + alpha * colored_mask
    return overlayed_image

test_image_path= '/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_trainable_fiducial.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
# for line in files:
#     img = line.split(' ')[0]
cleaned_image_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/circle/'
annotation_path = '/home/huifang/workspace/code/fiducial_remover/location_annotation/'
mask_path1 = '/home/huifang/workspace/code/backgroundremover/bgrm_direct_out/'
mask_path2 = '/home/huifang/workspace/code/backgroundremover/bgrm_out/'
iou1=0
iou2=0
cnt=0
for i in range(0,167):
    # img = image_path+str(i)+'.png'
    # print(files[i])
    # test = input()
    level = int(files[i].split(' ')[1])
    if level ==1:
        continue

    image_orig,h_scale,w_scale = get_image(files[i].split(' ')[0],return_ratio=True)

    cleaned_image = get_image(cleaned_image_path+str(i)+'.png')

    mask1 = get_mask(mask_path1+str(i)+'.png')
    mask2 = get_mask(mask_path2 + str(i) + '.png')

    ground_truth = read_labelme_json(annotation_path+str(i)+'.json',mask1.shape,[h_scale,w_scale])
    if np.max(ground_truth)==0:
        continue

    iou1 +=dice_coefficient(mask1, ground_truth)
    iou2 += dice_coefficient(mask2, ground_truth)
    cnt +=1
    #
    #
    overlayed_image1 = get_overlayed_image(image_orig,mask1)
    overlayed_image2 = get_overlayed_image(cleaned_image,mask2)
    overlayed_image3 = get_overlayed_image(image_orig,ground_truth)

    # Display the original image and the overlayed image
    plt.figure(figsize=(18, 18))

    plt.subplot(2,2, 1)
    plt.imshow(image_orig)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(overlayed_image3)
    plt.title('Tissue segmentation ground truth')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(overlayed_image1)
    plt.title('Tissue segmentation with fiducial markers')
    plt.axis('off')



    plt.subplot(2, 2, 4)
    plt.imshow(overlayed_image2)
    plt.title('Tissue segmentation without fiducial markers')
    plt.axis('off')

    plt.show()


    # image = io.imread(img)
    # # Apply a threshold to get a binary image
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # thresh = filters.threshold_otsu(gray)
    # binary = gray > thresh
    #
    # # Perform morphological operations to remove small noise
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
    #
    # # Background area determination
    # sure_bg = cv2.dilate(opening, kernel, iterations=3)
    #
    # # Foreground area determination
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    #
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    #
    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    #
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers + 1
    #
    # # Now, mark the region of unknown with zero
    # markers[unknown == 255] = 0
    #
    # # Apply the watershed algorithm
    # markers = cv2.watershed(image, markers)
    #
    # # Create an image to visualize the results
    # segmented_image = image
    #
    # # Coloring the segmented regions
    # for marker in np.unique(markers):
    #     if marker == -1:  # Boundary
    #         segmented_image[markers == marker] = [255, 0, 0]  # Red color for boundaries
    #     elif marker != 1:  # Not background
    #         segmented_image[markers == marker] = [0, 255, 0]  # Green color for objects
    #
    # # Display the result
    # io.imshow(segmented_image)
    # io.show()

print(iou1/cnt)
print(iou2/cnt)

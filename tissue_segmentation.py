import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import io, color, filters
# Load an H&E stained image


def find_nearest_multiple_of_32(x):
    base = 32
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)

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

def get_image(path):
    img_pil = Image.open(path)
    h, w = img_pil.size
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new), Image.ANTIALIAS)
    image = np.array(img_pil)
    # Normalize the image if necessary
    image = image / 255.0 if np.max(image) > 1 else image
    return image

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
mask_path1 = '/home/huifang/workspace/code/backgroundremover/bgrm_direct_out/'
mask_path2 = '/home/huifang/workspace/code/backgroundremover/bgrm_out/'
for i in range(101,167):
    print(i)


    # img = image_path+str(i)+'.png'
    # print(files[i])
    # test = input()
    level = int(files[i].split(' ')[1])
    if level ==1:
        continue

    image_orig = get_image(files[i].split(' ')[0])
    cleaned_image = get_image(cleaned_image_path+str(i)+'.png')


    mask1 = get_mask(mask_path1+str(i)+'.png')
    mask2 = get_mask(mask_path2 + str(i) + '.png')

    overlayed_image1 = get_overlayed_image(image_orig,mask1)
    overlayed_image2 = get_overlayed_image(cleaned_image,mask2)

    # Display the original image and the overlayed image
    plt.figure(figsize=(18, 6))

    plt.subplot(1,3, 1)
    plt.imshow(image_orig)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(overlayed_image1)
    plt.title('Tissue segmentation with fiducial markers')
    plt.axis('off')

    plt.subplot(1, 3, 3)
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
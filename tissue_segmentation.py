import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from skimage import io, color, filters
# Load an H&E stained image
from matplotlib.colors import LinearSegmentedColormap


import matplotlib.colors as mcolors
def read_labelme_json(json_file, image_shape, scale,label='tissue'):
    with open(json_file) as file:
        data = json.load(file)
    mask = np.zeros(image_shape[:2], dtype=np.float32)  # Assuming grayscale mask
    for shape in data['shapes']:
        if shape['label'] == label:
            polygon = np.array(shape['points'])
            polygon[: ,0] = polygon[:, 0]*scale[0]
            polygon[:, 1] = polygon[:, 1] * scale[1]
            polygon = np.asarray(polygon,dtype=np.int32)
            cv2.fillPoly(mask, [polygon], color=1)
    # mask = (mask > 128)
    return mask

def read_labelme_contour(json_file, scale, label='tissue'):
    contours = []
    with open(json_file) as file:
        data = json.load(file)
    for shape in data['shapes']:
        if shape['label'] == label:
            polygon = np.array(shape['points'])
            polygon[:, 0] *= scale[0]
            polygon[:, 1] *= scale[1]
            polygon = np.asarray(polygon, dtype=np.int32)

            # Append the first point to the end of the polygon to close the contour
            polygon = np.vstack([polygon, polygon[0]])
            polygon = gaussian_filter(polygon, sigma=0.3)

            contours.append(polygon)
    return contours

def find_nearest_multiple_of_32(x):
    base = 32
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)
def binarize_array(array, threshold):
    """
    Binarizes a numpy array based on a threshold determined by the given percentile.

    :param array: numpy array to be binarized
    :param percentile: percentile value used to determine the threshold, defaults to 50 (median)
    :return: binarized numpy array
    """
    binary_array = (array >= threshold).astype(int)

    return binary_array
def calculate_iou(mask1, mask2, tissue_value):
    mask1 = binarize_array(mask1,tissue_value)
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
    # threshold_value = 50  # Adjust the threshold value as needed
    # mask = (mask> threshold_value)

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
def get_overlay_ax(ax, image, mask):
    """
    Plots the RGB image with an overlaid mask on the provided axis.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        image (ndarray): RGB image array.
        mask (ndarray): Grayscale mask array.
    """
    # Set zero values in the mask to NaN so they won't be displayed

    # mask = mask.astype(float)  # Convert to float to allow NaN
    # mask = mask / 255
    # mask[mask < 0.2] = np.nan
    mask = (mask >= 0.2).astype(float)

    mask[mask == 0] = np.nan

    # Display the RGB image
    ax.imshow(image)
    # Define your custom color values as a list
    # color_values = ["#3f5d7d", "#ffcf4b", "#f2b134", "#f27649"]  # Example hex colors
    # colors = ["#f27649","#f2b134","#ffcf4b", "#3f5d7d"]
    # colors=['#501d8a','#1c8041','#e55709','#7e2f8c','#52bcec','#73aa43']
    # colors=['#c6d182','#eae0e9', '#e0c7e3', '#ae98b6', '#846e89']
    colors=['#d0b0c8','#b06098','#380850','blue']
    # # Create a ListedColormap
    # custom_cmap = mcolors.ListedColormap(color_values)

    # colors = ['#FEDACC', '#FD9778']
    # colors = ['#FEDACC','#FDD2C2','#FDCBB9','#FDC3B0','#FDBCA6','#FD6B48']
    # FEDACC
    # FDD2C2
    # FDCBB9
    # FDC3B0
    # FDBCA6
    # FDB49D
    # FDAD94
    # FDA58A
    # FD9E81
    # FD9778

    # Create the colormap
    # custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    custom_cmap = mcolors.ListedColormap(colors)
    # Overlay the grayscale mask with a colormap and transparency, ignoring zero (NaN) areas
    # overlay = ax.imshow(mask, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    overlay = ax.imshow(mask, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=1)

    # Add a color bar for the mask on the figure level
    cbar = plt.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mask Intensity")  # Label for the color bar

    # Remove axis ticks for a cleaner look
    ax.axis("off")


test_image_path= '/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_trainable_fiducial.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
# for line in files:
#     img = line.split(' ')[0]
visualization = True
plot = False
plot_bar=True
cleaned_image_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/application/model_out/recovery/'
annotation_path = '/home/huifang/workspace/code/fiducial_remover/location_annotation/'
mask_path1 = '/home/huifang/workspace/code/fiducial_remover/temp_result/application/bgrm-backup/'
mask_path2 = '/home/huifang/workspace/code/fiducial_remover/temp_result/application/bgrm-backup/'
iou1=0
iou2=0
cnt=0
for i in range(141,167):
    print(i)
    # print(cnt)
    # img = image_path+str(i)+'.png'
    # print(files[i])
    # test = input()

    level = int(files[i].split(' ')[1])
    # if level ==1:
    #     continue

    image_orig,h_scale,w_scale = get_image(files[i].split(' ')[0],return_ratio=True)

    cleaned_image = get_image(cleaned_image_path+str(i)+'.png')

    # try:
    #     mask1 = get_mask(mask_path1+str(i)+'_original2.png')
    # except:
    #     mask1 = get_mask(mask_path1 + str(i) + '_original2.png')
    # mask2 = get_mask(mask_path2 + str(i) + '.png')
    try:
        mask1 = get_mask(mask_path1+str(i)+'original2.png')
    except:
        mask1 = get_mask(mask_path1 + str(i) + '_original2.png')
    mask2 = get_mask(mask_path2 + str(i) + '.png')
    mask1 = mask1*1.0/np.max(mask1)
    mask2 = mask2 * 1.0 / np.max(mask2)

    ground_truth = read_labelme_json(annotation_path+str(i)+'.json',mask1.shape,[h_scale,w_scale])
    if np.max(ground_truth)==0:
        continue
    print(cnt)



    cnt +=1

    #
    if visualization:

        # Display the original image and the overlayed image
        fig, axs = plt.subplots(2, 2, figsize=(18, 18))

        # Use get_overlay_ax to plot on the first subplot
        axs[0,0].imshow(image_orig)

        img_gray = np.dot(image_orig[..., :3], [0.2989, 0.5870, 0.1140])

        # axs[0,1].imshow(img_gray, cmap="gray")  # Display the image in grayscale or other colormap as needed

        # Overlay the contours
        # ground_truth_contours = read_labelme_contour(annotation_path + str(i) + '.json', [h_scale, w_scale])
        # for contour1 in ground_truth_contours:
        #     # ax.plot(contour1[:, 0], contour1[:, 1], color="cyan", linewidth=1.5, label="Contour 1")
        #     axs[0,1].plot(contour1[:, 0], contour1[:, 1], color='blue', linewidth=3, label="Contour 1")
        # # Customize plot appearance
        # # plt.savefig('./temp_result/' + str(i) + '.png', dpi=600)
        # plt.axis("off")  # Hide axis if desired

        get_overlay_ax(axs[0, 1], image_orig, ground_truth)
        get_overlay_ax(axs[1, 0], image_orig, mask1)  # Plot overlay on the first subplot
        get_overlay_ax(axs[1, 1], cleaned_image, mask2)  # Plot overlay on the first subplot

        plt.show()
        save_path = './temp_result/vispro/'
        output_paths = [save_path+str(i)+"_1.png", save_path+str(i)+"_2.png", save_path+str(i)+"_3.png", save_path+str(i)+"_4.png"]
        for idx, ax in enumerate(axs.flat):
            # Turn off axis labels and ticks
            ax.axis('off')
            # Specify a bounding box that includes both the axis and the color bar
            extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(output_paths[idx], bbox_inches=extent, dpi=300)
        print('saved')
        test = input()


        # overlayed_image1 = get_overlayed_image(image_orig,mask1)
        # overlayed_image2 = get_overlayed_image(cleaned_image,mask2)
        # overlayed_image3 = get_overlayed_image(image_orig,ground_truth)
        #
        # # Display the original image and the overlayed image
        # plt.figure(figsize=(18, 18))
        #
        # plt.subplot(2,2, 1)
        # plt.imshow(image_orig)
        # plt.title('Original Image')
        # plt.axis('off')
        #
        # plt.subplot(2, 2, 2)
        # plt.imshow(overlayed_image3)
        # plt.title('Tissue segmentation ground truth')
        # plt.axis('off')
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(overlayed_image1)
        # plt.title('Tissue segmentation with fiducial markers')
        # plt.axis('off')
        #
        #
        #
        # plt.subplot(2, 2, 4)
        # plt.imshow(overlayed_image2)
        # plt.title('Tissue segmentation without fiducial markers')
        # plt.axis('off')
        #
        # plt.show()
        # Initialize lists to store IoU values
    if plot_bar:

        iou_original = calculate_iou(mask1, ground_truth, 0.4)
        iou_vispro = calculate_iou(mask2, ground_truth, 0.4)
        # Bar plot for comparison
        labels = ['Original image', 'Vispro']
        ious = [iou_original, iou_vispro]
        colors = ['royalblue', 'gold']

        plt.figure(figsize=(6, 6))
        bars = plt.bar(labels, ious, color=colors, width=0.5)

        # Annotate bars with their IoU values
        for bar, iou, color in zip(bars, ious, colors):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f"{iou:.2f}",
                     ha='center', va='bottom', fontsize=14, color='white', fontweight='bold')

        # Add labels and title
        plt.ylabel('IoU', fontsize=22)
        plt.ylim(0, 1.1)
        plt.title('Comparison of Best IoU Values', fontsize=22)
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.show()
    if plot:
        iou_original_values = []
        iou_vispro_values = []
        v_values = np.arange(0, 1, 0.1)

        # Calculate IoU for each value of v and store it
        for v in v_values:
            iou_original = calculate_iou(mask1, ground_truth, v)
            iou_vispro = calculate_iou(mask2, ground_truth, v)

            iou_original_values.append(iou_original)
            iou_vispro_values.append(iou_vispro)

        # Find the max values and corresponding v values for both
        max_iou_original = max(iou_original_values)
        max_v_original = v_values[np.argmax(iou_original_values)]

        max_iou_vispro = max(iou_vispro_values)
        max_v_vispro = v_values[np.argmax(iou_vispro_values)]

        # Plot the IoU values as v changes

        plt.figure(figsize=(6, 6))
        # plt.grid(True, axis='x')
        plt.grid(False)
        plt.plot(v_values, iou_original_values, label='Original image', marker='s', markersize=12, linestyle='-',
                 color='royalblue', linewidth=3)
        plt.plot(v_values, iou_vispro_values, label='Vispro', marker='^', markersize=12, linestyle='-', color='gold',
                 linewidth=3)

        # Add dashed lines for max points and annotate them
        plt.axhline(y=max_iou_original, color='royalblue', linestyle='--', linewidth=2)
        plt.axvline(x=max_v_original, color='royalblue', linestyle='--', linewidth=2)
        plt.text(max_v_original+0.08, max_iou_original-0.1, f"Max Original: {max_iou_original:.2f}", ha='center', va='bottom',
                 color='royalblue',fontsize=14)

        plt.axhline(y=max_iou_vispro, color='gold', linestyle='--', linewidth=2)
        plt.axvline(x=max_v_vispro, color='gold', linestyle='--', linewidth=2)
        plt.text(max_v_vispro-0.07, max_iou_vispro+0.01, f"Max Vispro: {max_iou_vispro:.2f}", ha='center', va='bottom',
                 color='gold',fontsize=14)

        # Add labels and title
        plt.rcParams['legend.fontsize'] = 14
        plt.xlabel('Tissue region threshold', fontsize=22)
        plt.ylabel('IoU', fontsize=22)
        plt.ylim(0, 1.1)
        plt.legend()
        # plt.savefig('./temp_result/vispro/' + str(i) + '_plot.png', dpi=600)
        plt.show()

        iou_original = calculate_iou(mask1, ground_truth, 0.5)
        iou_vispro = calculate_iou(mask2, ground_truth, 0.5)
        iou1 += iou_original
        iou2 += iou_vispro

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
print(cnt)
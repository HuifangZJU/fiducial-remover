import argparse
import statistics
import time
import torch
from backgroundremover.bg import remove
from omegaconf import OmegaConf
import yaml
from scipy.ndimage import zoom
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
from cnn_io import *
from hough_utils import *
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy import ndimage
import fiducial_utils
from multiprocessing import Pool, cpu_count
import scipy.ndimage as ndi
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
Image.MAX_IMAGE_PIXELS = 3000000000
import cv2
def get_image_mask_from_annotation(image_size,annotation,step):
    image_mask = np.zeros(image_size)
    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            patch_x = i * step
            patch_y = j * step
            image_mask[patch_x:patch_x + step, patch_y:patch_y + step] = annotation[i, j]
    return image_mask



def apply_gaussian_kernel(image, sigma=1.0):
    """
    Apply Gaussian kernel to the input image

    :param image: Input image (numpy ndarray)
    :param sigma: Standard deviation of the Gaussian kernel
    :return: Image after applying Gaussian kernel
    """
    return gaussian_filter(image, sigma=sigma)

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def get_masked_image(image,circle):
    for c in circle:
        cv2.circle(image, (c[0], c[1]), c[2], 1,-1)
    return image

def refine_frame_mask(mask, kernel_size=1):
    """
    Refines the input mask by applying morphological opening to remove small noise.

    :param mask: np.ndarray, binary image mask
    :param kernel_size: int, size of the morphological kernel
    :return: np.ndarray, refined binary image mask
    """
    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform morphological opening
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return refined_mask

def remove_small_objs(mask,size_threshold):
    # Label connected components
    label_im, nb_labels = ndimage.label(mask)

    # Find the sizes of the connected components
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    # Create a mask to remove small components
    mask_size = sizes < size_threshold
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0

    # The final mask without small noisy components
    mask_cleaned = label_im > 0
    return mask_cleaned

def binarize_array(array, threshold):
    """
    Binarizes a numpy array based on a threshold determined by the given percentile.

    :param array: numpy array to be binarized
    :param percentile: percentile value used to determine the threshold, defaults to 50 (median)
    :return: binarized numpy array
    """
    # Compute the threshold as the specified percentile of the array
    # threshold = np.percentile(array, percentile)

    # Binarize the array based on the threshold
    binary_array = (array >= threshold).astype(int)

    return binary_array

def get_image_var(image_name):
    img_pil = Image.open(image_name.split(' ')[0])
    h, w = img_pil.size
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new), Image.ANTIALIAS)

    img_np = np.array(img_pil)
    img_var = transforms_rgb(img_pil)
    img_var = torch.unsqueeze(img_var, dim=0).to(device)
    return img_var,img_np

def get_cnn_mask(img_var,circle_generator):
    cnn_mask_var = circle_generator(img_var)
    cnn_mask = cnn_mask_var.cpu().detach().numpy().squeeze()
    cnn_mask = np.transpose(cnn_mask, (0, 1))
    cnn_mask = normalize_array(cnn_mask)
    return cnn_mask

def get_position_mask(img_var,position_generator, use_gaussian=True):
    position_var = position_generator(img_var)
    position = position_var.cpu().detach().numpy().squeeze()
    position = np.transpose(position, (0, 1))

    if use_gaussian:
        position = apply_gaussian_kernel(position, sigma=1.5)
    _, _, w, h = img_var.shape
    position = get_image_mask_from_annotation([w, h], position, patch_size)
    position = normalize_array(position)
    return position.astype(np.float32)

def get_circle_and_position_mask(img_var,generator):
    cnn_mask_var, position_var,attn_var = generator(img_var)
    cnn_mask = cnn_mask_var.cpu().detach().numpy().squeeze()
    cnn_mask = np.transpose(cnn_mask, (0, 1))
    cnn_mask = binarize_array(cnn_mask, 0.5)
    return cnn_mask


def morphological_closing(binary_mask):
    # Perform Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=7)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Perform the morphological opening
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    return opened_mask

def get_binary_mask(network_mask):
 #   Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(network_mask, (3, 3), 0)

    threshold = 0.2  # This can be adjusted based on your observations
    _, binary_mask = cv2.threshold(blurred_mask, threshold, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    return binary_mask

def get_cnn_position_mask(cnn_mask,position):

    cnn_position_mask = cnn_mask*position
    # cnn_position_mask = cnn_mask
    binary_mask = get_binary_mask(cnn_position_mask)
    # final_mask = morphological_closing(binary_mask)

    # plt.imshow(1 - binary_mask, cmap='gray')
    # plt.show()
    return binary_mask


def get_inpainting_result(inpainter_image,mask):
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    inpainter_image_var = torch.from_numpy(inpainter_image).unsqueeze(0).to(device)

    batch = dict(image=inpainter_image_var, mask=mask)
    with torch.no_grad():
        batch = inpainter(batch)
    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    unpad_to_size = batch.get('unpad_to_size', None)
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

def remove_bg(src_img_path, out_img_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    f = open(src_img_path, "rb")
    data = f.read()
    img = remove(data, model_name=model_choices[0],
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10,
                 alpha_matting_erode_structure_size=10,
                 alpha_matting_base_size=1000)
    f.close()
    img[0].save(out_img_path)
def get_rgb_img(path,out_path):
    # Open the RGBA image
    rgba_image = Image.open(path).convert("RGBA")

    # Convert RGBA image to a NumPy array
    rgba_array = np.array(rgba_image)

    # Extract the alpha channel
    alpha_channel = rgba_array[..., 3]

    # Set the alpha channel to binary: fully transparent (0) or fully opaque (255)
    binary_alpha = np.where(alpha_channel > 200, 255, 0).astype(np.uint8)

    # Replace the original alpha channel with the binary alpha
    rgba_array[..., 3] = binary_alpha



    # Convert the modified NumPy array back to a PIL RGBA image
    binary_rgba_image = Image.fromarray(rgba_array, 'RGBA')

    # Create a white background image (RGBA)
    white_background = Image.new('RGBA', binary_rgba_image.size, (255, 255, 255, 255))

    # Blend the binary RGBA image with the white background
    blended_image = Image.alpha_composite(white_background, binary_rgba_image).convert('RGB')

    # Convert the blended image to a NumPy array and return
    rgb_array = np.array(blended_image)
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(rgb_array)

    # Save the image as a PNG
    img.save(out_path)


def transform_points(points, tx, ty, scale_x, scale_y, angle):
    scaling_matrix = np.array([[scale_x, 0], [0, scale_y]])
    rotation_matrix = R.from_euler('z', angle, degrees=True).as_matrix()[:2, :2]
    translation_vector = np.array([tx, ty])

    transformed_points = np.dot(points, scaling_matrix)
    transformed_points = np.dot(transformed_points, rotation_matrix) + translation_vector
    return transformed_points

def split_array(arr, cond):
  return arr[cond], arr[~cond]


def process_and_visualize_mask(mask):
    # Find the smallest rectangle that contains all the 1's
    y_indices, x_indices = np.where(mask == 1)
    x_min, x_max, y_min, y_max = np.min(x_indices), np.max(x_indices), np.min(y_indices), np.max(y_indices)

    # Crop the mask to the smallest rectangle
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1].copy()

    # Erode the mask incrementally and find the largest rectangle that contains only 0's
    step = 16 # Define step size for eroding
    last_non_empty = cropped_mask.copy()
    cnt=0
    while np.any(cropped_mask):
        last_non_empty = cropped_mask.copy()
        cropped_mask = last_non_empty[step:-step,step:-step]
        cnt+=1
    offset = step * cnt
    # inner_x_min, inner_x_max, inner_y_min, inner_y_max = (
    #     x_min+offset,
    #     x_max-offset,
    #     y_min+offset,
    #     y_max-offset,
    # )
    rec1=(x_min,y_min,x_min+offset,y_max-offset)
    rec2 = (x_min+offset, y_min, x_max, y_min + offset)
    rec3 = (x_max-offset, y_min+offset, x_max, y_max)
    rec4 = (x_min, y_max-offset, x_max-offset, y_max)
    # # Visualization
    # plt.imshow(mask, cmap='gray')
    # # Outer rectangle
    # plt.gca().add_patch(
    #     patches.Rectangle((x2min, y2min), x2max - x2min + 1, y2max - y2min + 1, linewidth=2, edgecolor='r',
    #                       facecolor='none'))
    # # Inner rectangle
    # plt.gca().add_patch(patches.Rectangle((x1min, y1min), x1max - x1min + 1,
    #                                       y1max - y1min + 1, linewidth=2, edgecolor='g', facecolor='none'))
    # # Outer rectangle
    # plt.gca().add_patch(
    #     patches.Rectangle((x3min, y3min), x3max - x3min + 1, y3max - y3min + 1, linewidth=2, edgecolor='r',
    #                       facecolor='none'))
    # # Inner rectangle
    # plt.gca().add_patch(patches.Rectangle((x4min, y4min), x4max - x4min + 1,
    #                                       y4max - y4min + 1, linewidth=2, edgecolor='g', facecolor='none'))
    #
    # plt.show()
    return [rec1,rec2,rec3,rec4]

def save_rgb_image(array, filename):
    # The input is expected to be in the range [0, 1], so we scale it to [0, 255]
    if np.max(array)<1.1:
        array = 255*array
    array = array.astype('uint8')
    cv2.imwrite(filename, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR format

# For arrays with shape [m, n]
def save_gray_image(array, filename):
    # The input is expected to be in the range [0, 1], so we scale it to [0, 255]
    array = (array * 255).astype('uint8')
    cv2.imwrite(filename, array)


def calculate_normalized_iou(mask1, mask2):
    # Calculate intersection and union
    mask1 = binarize_array(mask1,0.5)
    mask2 = binarize_array(mask2, 0.5)
    #
    # f,a = plt.subplots(1,2)
    # a[0].imshow(~mask1)
    # a[1].imshow(~mask2)
    # plt.show()

    # Calculate Intersection and Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 1.0  # Avoid division by zero

    return iou
    # return average_iou
def resize_binary_mask(binary_mask, target_shape):
    # Calculate scaling factors for resizing
    scale_factors = (np.array(target_shape) / np.array(binary_mask.shape)).tolist()

    # Resize the binary mask using nearest-neighbor interpolation
    resized_mask = zoom(binary_mask, scale_factors, order=0)

    # Threshold the resized mask to maintain binary values
    resized_mask[resized_mask >= 0.5] = 1
    resized_mask[resized_mask < 0.5] = 0

    return resized_mask


def divide_cytassist_and_process(image_name):
    """
    Divide the image into 4 patches, process each patch, and reassemble the image.
    """
    img_pil = Image.open(image_name)
    h, w = img_pil.size
    h_new = find_nearest_multiple_of_64(h)
    w_new = find_nearest_multiple_of_64(w)
    img_pil = img_pil.resize((h_new, w_new), Image.ANTIALIAS)
    img_np = np.array(img_pil)


    img_var = transforms_rgb(img_pil)
    img_var = torch.unsqueeze(img_var, dim=0).to(device)
    mid_h, mid_w = h_new // 2, w_new // 2
    # Divide the image into 4 patches
    var_patches = [img_var[:,:,:mid_h, :mid_w], img_var[:,:,:mid_h, mid_w:],
               img_var[:,:,mid_h:, :mid_w], img_var[:,:,mid_h:, mid_w:]]
    np_patches=[img_np[:mid_h, :mid_w,:],img_np[:mid_h, mid_w:,:],
                img_np[mid_h:, :mid_w,:], img_np[mid_h:, mid_w:,:]]

    # Process each patch and store the 5 results
    results = [run(var_patch,np_patch) for var_patch,np_patch in zip(var_patches,np_patches)]

    # Stack results of the same type from each patch
    final_results = []
    for i in range(5):
        # Extract the i-th result from each patch's results
        combined_patches = [result[i] for result in results]

        # Combine the patches for this particular result
        top_combined = np.hstack((combined_patches[0], combined_patches[1]))
        bottom_combined = np.hstack((combined_patches[2], combined_patches[3]))
        combined_image = np.vstack((top_combined, bottom_combined))

        final_results.append(combined_image)
    return img_np,final_results


def run(image_var,image_np,recovery=True):
    cnn_mask, position = get_circle_and_position_mask(image_var,generator,use_gaussian=True)


    # position = binarize_array(position,0.6)
    # cnn_mask = get_binary_mask(cnn_mask)
    # position = get_binary_mask(position)
    # plt.imshow(cnn_mask)
    # plt.show()
    # continue

    # binary_position = binarize_array(position,0.5)
    # plt.imshow(binary_position)
    # plt.show()
    # position_square_params = fit_square_to_mask(position)
    # cx, cy, s = position_square_params
    # s = abs(s)

    # single_cnn_position_mask = get_cnn_position_mask(single_cnn_mask ,position)
    cnn_position_mask = get_cnn_position_mask(cnn_mask, position)
    # cnn_mask_circles, cnn_mask_circle_figure, radius = run_geometric(cnn_mask,position, run_square=False,
    #                                                                   return_radius=True)
    # cnn_mask=cnn_mask_circle_figure
    # plt.imshow(cnn_mask_circle_figure)
    # plt.show()
    # cnn_mask = cnn_mask_circle_figure

    if recovery:
        inpainter_image = np.transpose(image_np,(2,0,1))
        inpainter_image = inpainter_image.astype('float32')/255
        cnn_position_output = get_inpainting_result(inpainter_image,cnn_position_mask)


        single_cnn_output = get_inpainting_result(inpainter_image,cnn_mask)
        # single_cnn_mask = get_binary_mask(cnn_mask_circle_figure)
        # direct_output = get_inpainting_result(inpainter_image,single_cnn_mask)
        return cnn_mask, position, cnn_position_mask, cnn_position_output, single_cnn_output
    else:
        return cnn_mask, position, cnn_position_mask

def plot_circles_in_image(image,circles,width):
    img_cp = image.copy()
    for circle in circles:
        cv2.circle(img_cp, (circle[0],circle[1]), circle[2],[255,0,0], width)

    return img_cp


def divide_image(image, mask, patch_size):
    """
    Divide the image and mask into smaller patches, handling padding if necessary.
    """
    c, h, w = image.shape[0], image.shape[1], image.shape[2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # Pad the image and mask
    padded_image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    img_patches = []
    mask_patches = []
    positions = []
    padded_h, padded_w = padded_image.shape[-2], padded_image.shape[-1]

    for i in range(0, padded_h, patch_size):
        for j in range(0, padded_w, patch_size):
            img_patch = padded_image[:, i:i + patch_size, j:j + patch_size]
            mask_patch = padded_mask[i:i + patch_size, j:j + patch_size]
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)
            positions.append((i, j))

    return img_patches, mask_patches, positions, (c,h, w)


def stitch_patches_incremental(results, positions, original_shape):
    """
    Stitch the patches back into a single image incrementally to reduce memory usage.
    """
    c, h, w = original_shape
    stitched_image = np.zeros((c, h, w), dtype=results[0].dtype)

    for patch, (i, j) in zip(results, positions):
        patch_c, patch_h, patch_w = patch.shape
        stitched_image[:, i:i + patch_h, j:j + patch_w] = patch[:, :min(patch_h, h - i), :min(patch_w, w - j)]

    return stitched_image



def run_single_tiff(tiff_image_path,high_res_image_path):


    tiff_image = plt.imread(tiff_image_path)
    img_var, _ = get_image_var(high_res_image_path)
    cnn_mask = get_circle_and_position_mask(img_var, generator)
    zoom_factors = (tiff_image.shape[0] / cnn_mask.shape[0], tiff_image.shape[1] / cnn_mask.shape[1])
    cnn_mask = zoom(cnn_mask, zoom_factors, order=0)
    # rgba images
    if tiff_image.shape[2] == 4:
        tiff_image = tiff_image[:, :, :3]

    inpainter_image = np.transpose(tiff_image, (2, 0, 1))
    inpainter_image = inpainter_image.astype('float32') / 255

    mid_h, mid_w = tiff_image.shape[0] // 2, tiff_image.shape[1] // 2
    patch_size = 3000  # Adjust this value as needed to get more patches
    img_patches, mask_patches, positions, original_shape = divide_image(inpainter_image, cnn_mask, patch_size)
    print(len(img_patches))
    # Process each patch and store the results
    results = []
    i = 0
    for img_patch, mask_patch in zip(img_patches, mask_patches):
        print(i)
        if np.any(mask_patch == 1):
            # Process through inpainting network if mask contains 1s
            result = get_inpainting_result(img_patch, mask_patch)
            result = np.transpose(result, (2, 0, 1))
            i = i + 1
        else:
            # Use the original image patch if mask does not contain 1s
            result = img_patch
            result = np.clip(result * 255, 0, 255).astype('uint8')

        results.append(result)

    # Stitch the processed patches back together incrementally
    stitched_result = stitch_patches_incremental(results, positions, original_shape)
    stitched_result = np.transpose(stitched_result, (1, 2, 0))
    # print(stitched_result.shape)
    pil_image = Image.fromarray(stitched_result)
    # print(pil_image.mode)
    #
    # plt.imshow(stitched_result)
    # plt.show()

    pil_image.save(tiff_image_path[:-4] + '_recovered.png')

    remove_bg(tiff_image_path[:-4] + '_recovered.png', tiff_image_path[:-4] + '_cleaned.png')
    get_rgb_img(tiff_image_path[:-4] + '_cleaned.png', tiff_image_path[:-4] + '_cleaned_with_bg.png')


# ------ arguments handling -------
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--pack_size', type=int, default=4, help='size of deep a image training patches')
parser.add_argument('--img_height', type=int, default=32, help='size of image height')
parser.add_argument('--img_width', type=int, default=32, help='size of image width')
parser.add_argument('--channel', type=int, default=3, help='number of image channel')
args = parser.parse_args()
os.makedirs('./test/', exist_ok=True)

# ------ device handling -------
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------ Configure model -------
generator = get_combined_Generator()
generator.eval()
generator.to(device)
# Initial image inpainter
inpainter_model_path = '/home/huifang/workspace/code/lama/big-lama'
train_config_path = inpainter_model_path+'/config.yaml'
with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

train_config.training_model.predict_only = True
train_config.visualizer.kind = 'noop'
inpainter = getLamaInpainter(train_config,inpainter_model_path+'/models/best.ckpt', strict=False, map_location='cpu')
inpainter.freeze()
inpainter.to(device)
# ------ main process -------
# manage input
transforms_rgb = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


imglist= "/media/huifang/data/fiducial/tiff_data/data_list.txt"
file = open(imglist)
lines = file.readlines()
num_files = len(lines)
for i in range(11,num_files):
    print(i)
    line = lines[i]
    line = line.rstrip().split(' ')
    tiff_image_path = line[0]

    high_res_image_path = line[2]
    run_single_tiff(tiff_image_path,high_res_image_path)


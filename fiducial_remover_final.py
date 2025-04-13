import argparse
import statistics
from omegaconf import OmegaConf
import yaml
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
from cnn_io import *
from hough_utils import *
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from backgroundremover.bg import remove
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from skimage.color import label2rgb
from skimage.transform import resize
from PIL import Image
import imageio
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

def get_rgb_img(rgba_image):

    # Create a new background image (white) with the same size as the RGBA image
    background = Image.new('RGBA', rgba_image.size, (255, 255, 255, 255))  # RGBA with white background

    # Blend the RGBA image with the white background
    blended_image = Image.alpha_composite(background, rgba_image).convert('RGB')

    # Convert the blended RGB image to a NumPy array
    rgb_array = np.array(blended_image)
    return rgb_array

def segregate(rgb_image,binary_mask):
    # rgb_image = get_rgb_img(img)
    # img = np.asarray(img)
    # binary_mask = img[:, :, 3].copy()
    # plt.imshow(binary_mask)
    # plt.show()
    # binary_mask[img[:, :, 3] > 50] = 1
    # binary_mask[img[:, :, 3] < 50] = 0
    # threshold_value = np.percentile(binary_mask, 95)
    # print(threshold_value)
    # test = input()

    # Apply thresholding to create a binary mask
    # _, binary_mask = cv2.threshold(binary_mask, 5, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary_mask)
    # plt.show()



    binary_mask = resize(binary_mask, rgb_image.shape[:2], order=0, preserve_range=True, anti_aliasing=False).astype(
        np.uint8)
    # plt.imshow(binary_mask)
    # plt.show()

    # Find connected components
    labeled_mask, num_features = label(binary_mask)
    # print(f"Number of disconnected components: {num_features}")
    size_threshold = 2000  # Adjust this threshold based on your requirements

    # Find slices for each connected component
    object_slices = find_objects(labeled_mask)

    # Create a mask to keep track of the components to remove
    mask_to_remove = np.zeros_like(labeled_mask, dtype=bool)

    for i, slice_tuple in enumerate(object_slices):
        if slice_tuple is not None:
            # Calculate the size of the component
            component_size = np.sum(labeled_mask[slice_tuple] == (i + 1))

            # Mark small components for removal
            if component_size < size_threshold:
                mask_to_remove[slice_tuple] |= (labeled_mask[slice_tuple] == (i + 1))

    # Remove small components by setting them to background (0)
    labeled_mask[mask_to_remove] = 0

    # Re-label the connected components
    labeled_mask, num_features = label(labeled_mask > 0)
    print(f"Number of disconnected components: {num_features}")

    # Overlay labeled mask on RGB image
    overlay = label2rgb(labeled_mask, image=rgb_image, bg_label=0, alpha=0.5, kind='overlay')
    # Convert the overlay image to an RGBA image (adds an alpha channel)
    rgba_overlay = np.zeros((overlay.shape[0], overlay.shape[1], 4), dtype=overlay.dtype)
    rgba_overlay[..., :3] = overlay
    rgba_overlay[..., 3] = (labeled_mask > 0).astype(overlay.dtype) * 1.0  # Alpha channel
    # rgba_overlay = rgba_overlay.astype(np.uint8)
    return rgba_overlay,num_features

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

def get_circle_and_position_mask(img_var,generator,use_gaussian=True):
    cnn_mask_var, position_var,attn_var = generator(img_var)
    # cnn_mask_var = generator(img_var)
    cnn_mask = cnn_mask_var.cpu().detach().numpy().squeeze()
    cnn_mask = np.transpose(cnn_mask, (0, 1))
    cnn_mask = binarize_array(cnn_mask, 0.5)

    position = position_var.cpu().detach().numpy().squeeze()
    position = np.transpose(position, (0, 1))
    if use_gaussian:
        position = apply_gaussian_kernel(position, sigma=0.8)
    _, _, w, h = img_var.shape
    position = get_image_mask_from_annotation([w, h], position, patch_size)
    position = normalize_array(position)

    # position = normalize_array(cnn_mask)
    return cnn_mask,position


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
    batch = inpainter(batch)
    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    unpad_to_size = batch.get('unpad_to_size', None)
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


def fit_points_to_square(points):

    hull = cv2.convexHull(points)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    center, size, angle = rect
    width, height = size
    half_side = min(width, height) / 2

    square = np.array([
        [center[0] - half_side, center[1] - half_side],
        [center[0] + half_side, center[1] - half_side],
        [center[0] + half_side, center[1] + half_side],
        [center[0] - half_side, center[1] + half_side]
    ])

    # Calculate the centroid of the box and square
    box_center = np.mean(box, axis=0)
    square_center = np.mean(square, axis=0)

    # Calculate the center of the resulting square
    avg_center = (box_center + square_center) / 2

    # Calculate the side length of the resulting square
    avg_side = np.mean([np.linalg.norm(vertex - avg_center) for vertex in np.vstack((box, square))]) * np.sqrt(2)

    central_square = np.array([
        [avg_center[0] - avg_side / 2, avg_center[1] - avg_side / 2],
        [avg_center[0] + avg_side / 2, avg_center[1] - avg_side / 2],
        [avg_center[0] + avg_side / 2, avg_center[1] + avg_side / 2],
        [avg_center[0] - avg_side / 2, avg_center[1] + avg_side / 2]
    ])
    return central_square
def run_geometric(img,position,run_square=True,return_radius=False):
    circles = run_circle_threhold(img, 10, circle_threshold=20, step=5)
    re_r = statistics.mode(circles[:,2])
    circles = run_circle_threhold(img, re_r, circle_threshold=int(2*re_r), step=3)
    new_circles=[]
    for x,y,r in circles:
        # if position[y,x]>0.1:
        new_circles.append([x,y,r+2])
    circles = np.asarray(new_circles)
    hough_mask = np.zeros(img.shape[:2])
    hough_mask = get_masked_image(hough_mask, circles)
    if run_square:
        central_square = fit_points_to_square(circles[:, :2])
        if return_radius:
            return circles, hough_mask, central_square,re_r
        else:
            return circles,hough_mask,central_square
    else:
        if return_radius:
            return circles, hough_mask,re_r
        else:
            return circles, hough_mask


def objective_function(params, mask):
    cx, cy, s = params
    s = abs(s)  # Ensure positive side length
    coords = np.array([
        [cx - s / 2, cy - s / 2],
        [cx + s / 2, cy - s / 2],
        [cx + s / 2, cy + s / 2],
        [cx - s / 2, cy + s / 2]
    ])

    num_points = 100
    edge_points = []
    for i in range(4):
        x_vals = np.linspace(coords[i, 0], coords[(i + 1) % 4, 0], num_points)
        y_vals = np.linspace(coords[i, 1], coords[(i + 1) % 4, 1], num_points)
        edge_points.append(np.vstack([x_vals, y_vals]))
    edge_points = np.hstack(edge_points)


    values = map_coordinates(mask, edge_points, order=1, mode='constant')
    return -np.mean(values)

def fit_square_to_mask(mask):
    kernel_size = 5  # Adjust based on your requirement
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    pad_size = 10  # Adjust as needed

    # Perform padding
    padded_mask = cv2.copyMakeBorder(mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)

    # Perform erosion on padded mask
    eroded_mask = cv2.erode(padded_mask, kernel, iterations=3)

    # Optionally, remove padding from the eroded mask if required
    eroded_mask = eroded_mask[pad_size:-pad_size, pad_size:-pad_size]
    # plt.imshow(eroded_mask)
    # plt.show()

    # Threshold the mask
    threshold = 0.5  # or any value that you find suitable
    _, thresh_mask = cv2.threshold(eroded_mask, threshold, 0.5, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours((thresh_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found!")

    # Get a bounding box around the detected contour
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Set initial parameters
    initial_params = [x + w / 2, y + h / 2, min(w, h)]  # corrected the side length

    # Run the optimization
    # Use a Gaussian kernel to weigh the contributions of neighboring pixels
    sigma = 1.5  # Standard deviation of the Gaussian kernel, adjust as needed
    gaussian_mask = gaussian_filter(mask, sigma=sigma)
    result = minimize(objective_function, initial_params, args=(gaussian_mask,), method='Nelder-Mead')
    cx, cy, s = result.x
    return [cx, cy, s]


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

    # cnn_mask = morphological_closing(get_binary_mask(get_cnn_mask(img_var)))
    # single_cnn_mask = get_cnn_mask(img_var,circle_generator)
    # position = get_position_mask(img_var,use_gaussian=True)

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
# # Initialize circle generator
# circle_generator = get_circle_Generator()
# circle_generator.to(device)
# # Initialize position generator
# position_generator = get_position_Generator()
# position_generator.to(device)
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
patch_size = 32
transforms_rgb = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# test_image_path = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
# test_image_path = '/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_trainable_fiducial.txt'
# test_image_path = '/home/huifang/workspace/data/imagelists/st_cytassist.txt'
# test_image_path='/home/huifang/workspace/data/imagelists/st_auto_trainable_images.txt'
# test_image_path='/home/huifang/workspace/data/imagelists/st_auto_test_images.txt'
test_image_path = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
# test_image_path='/home/huifang/workspace/data/imagelists/st_image_with_aligned_fiducial.txt'
#
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
num_files = len(files)
fiducial_ious_cnn=0
fiducial_ious_cnn_position=0
binary_ious=0
cnt=0
Visualization = True
for i in range(28,num_files):
    print(str(num_files)+'---'+str(i))
    start_time = time.time()
    image_name = files[i]
    image_name = "/home/huifang/workspace/code/fiducial_remover/location_annotation/tissue_hires_image.png"
    # directory, _ = os.path.split(image_name.split(' ')[0])
    # aligned_path = directory+'/aligned_fiducials.jpg'
    # label_percentage = float(image_name.split(' ')[1])
    # if label_percentage >0.9:
    #     continue
    # level = int(image_name.split(' ')[1])
    # if level ==1:
    #     continue

    # group = int(image_name.split(' ')[2])
    # if group!=1:
    #     continue
    # img_pil= Image.open(image_name.split(' ')[0])
    # img_tissue = plt.imread(image_name.split(' ')[1].rstrip('\n'))
    # img_np,[cnn_mask, position, cnn_position_mask, cnn_position_output, single_cnn_output] = divide_cytassist_and_process(image_name.split(' ')[0])

    # if not os.path.exists(image_name.split(' ')[0].split('.')[0] + '_10x.png'):
    #     continue
    # mask_10x = plt.imread(image_name.split(' ')[0].split('.')[0] + '_10x.png')
    # circle_gt = plt.imread(image_name.split(' ')[0].split('.')[0] + '_ground_truth.png')


    img_var,img_np = get_image_var(image_name.split(' ')[0])

    # mask_10x = resize_binary_mask(mask_10x, circle_gt.shape[:2])
    # binary_gt = plt.imread(image_name.split('.')[0]+ '_binary_gt.png')
    # binary_gt = plt.imread(image_name.split('.')[0] + '_binary_patch.png')
    # inpainter_image = np.transpose(img_np, (2, 0, 1))
    # inpainter_image = inpainter_image.astype('float32') / 255
    # output_10x = get_inpainting_result(inpainter_image, mask_10x)
    # plt.imshow(output_10x)
    # plt.show()

    # aligned_image = plt.imread(aligned_path)
    # save_rgb_image(output_10x, './10x_result/recovery/' + str(i) + '.png')
    # save_rgb_image(aligned_image, './10x_result/alignment/' + str(i) + '.png')
    # continue
    # f,a = plt.subplots(1,3)
    # aligned_image = plt.imread(image_name.split(' ')[1].rstrip('\n'))
    # a[0].imshow(aligned_image)
    # a[1].imshow(img_np)
    # a[1].imshow(1-mask_10x,cmap='binary',alpha=0.6)
    # a[2].imshow(output_10x)
    #
    # plt.show()
    # test = input()

    # if not os.path.exists(image_name.split('.')[0] + '_auto.npy'):
    #     continue
    # circles = np.load(image_name.split('.')[0] + '_10x.npy')
    # image_original = plt.imread(image_name.split(' ')[0])
    # img_auto = plot_circles_in_image(img_original, circles, 2)


    cnn_mask,position,cnn_position_mask,cnn_position_output,single_cnn_output = run(img_var,img_np)
    # cleaned_img,mask = remove_bg(single_cnn_output)
    plt.imshow(cnn_mask)
    plt.show()

    plt.imshow(single_cnn_output)
    plt.show()
    pil_image = Image.fromarray(single_cnn_output)
    pil_image.save(image_name.split(' ')[0][:-4] + '_recovered.png')

    remove_bg(image_name.split(' ')[0][:-4] + '_recovered.png', image_name.split(' ')[0][:-4] + '_cleaned.png')

    get_rgb_img(image_name.split(' ')[0][:-4] + '_cleaned.png', image_name.split(' ')[0][:-4] + '_cleaned_with_bg.png')
    fragments,num_tissue = segregate(cleaned_img,mask)

    plt.imshow(fragments)
    plt.show()






    save_rgb_image(cleaned_img,'./temp_result/application/bgrm/' + str(i) + '.png')
    # cleaned_img.save('./temp_result/application/bgrm/' + str(i) + '.png')
    # save_rgb_image(fragments, './temp_result/application/fragments/' + str(i) + '_'+str(num_tissue)+'.png')
    imageio.imwrite('./temp_result/application/fragments/' + str(i) + '_'+str(num_tissue)+'.png', fragments)
    test = input()
    continue
    # # cnn_mask, position, cnn_position_mask = run(img_var, img_np,recovery=False)
    #
    image_original = plt.imread(image_name.split(' ')[0])*255
    image_original = np.asarray(image_original,np.uint8)

    # mask_10x = generate_mask(image_original.shape[:2], circles, -1)
    # cnn_mask = cv2.resize(cnn_mask, (image_original.shape[1],image_original.shape[0]), interpolation=cv2.INTER_NEAREST)
    # cnn_mask  = np.asarray(mask_10x,np.uint8)


    # cnn_mask = remove_small_objs(cnn_mask, 100)
    # f,a = plt.subplots(1,2)
    # a[0].imshow(cnn_mask)
    # a[1].imshow(cnn_mask_cleaned)
    # plt.show()

    # mask_bool = cnn_mask.astype(bool)
    #
    # # Create an all-zero image with the same shape as the original image
    # green_mask = np.zeros_like(img_np)
    #
    # # Wherever the mask is True, set the color to green
    # green_mask[mask_bool] = [255, 0, 0]
    #
    # # Blend the green mask with the original image
    # # You can adjust the transparency by changing alpha (0 - transparent, 1 - opaque)
    # alpha = 0.5
    # overlay_image = cv2.addWeighted(green_mask, alpha, img_np, 0.5, 0)
    #
    # # plt.imshow(overlay_image)
    # # plt.show()
    #
    #
    # # kernel = np.ones((5, 5), np.uint8)
    # # cnn_mask = cv2.erode(cnn_mask, kernel, iterations=1)
    # # contours, hierarchy = cv2.findContours(cnn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(image_original, contours, -1, color=255, thickness=2)
    # save_rgb_image(overlay_image, '/home/huifang/workspace/code/fiducial_remover/10x_result/mask/' + str(i) + '.png')
    # # save_rgb_image(image_original, './temp_result/method/our_output_without_spatial/all/' + str(i) + '.png')
    # continue

    # position = plt.imread(image_name.split('.')[0] + '_binary_patch.png')
    # plt.imshow(binarize_array(position,0.5))
    # plt.show()

    # test_circle_mask_cnn_position = resize_binary_mask(cnn_position_mask, circle_gt.shape)
    cnn_mask= resize_binary_mask(cnn_mask, circle_gt.shape)
    # test_binary_mask = resize_binary_mask(position,binary_gt.shape)
    #
    fiducial_iou_cnn = calculate_normalized_iou(circle_gt,cnn_mask)
    fiducial_ious_cnn += fiducial_iou_cnn
    # fiducial_iou_cnn_position = calculate_normalized_iou(circle_gt, test_circle_mask_cnn_position)
    # fiducial_ious_cnn_position += fiducial_iou_cnn_position
    # binary_iou = calculate_normalized_iou(binary_gt,test_binary_mask)
    # binary_ious+=binary_iou
    # print(fiducial_iou_cnn)
    cnt+=1

    end_time = time.time()
    # save_gray_image(cnn_mask,'./temp_result/method/attn_net_output/train/'+str(i)+'.png')
    # save_rgb_image(single_cnn_output,'./temp_result/method/attn_net_output/train/'+str(i)+'.png')
    # # save_rgb_image(cnn_position_output, './temp_result/cnn_mul_position/' + str(i) + '.png')
    # continue
    if Visualization:
        f,a = plt.subplots(1,4,figsize=(20, 10))
        a[0].imshow(img_np)
        # a[0,1].imshow(img_tissue)
        # a[1].imshow(img_np)
        # a[1].imshow(1 - cnn_mask, cmap='binary', alpha=0.6)
        # a[0,1].imshow(img_auto)
        # a[0,1].imshow(1 - auto_mask, cmap='binary', alpha=0.6)
        # a[0,1].imshow(1-circle_gt,cmap='gray')
        a[1].imshow(single_cnn_output)
        # a[0,2].imshow(1-cnn_mask,cmap='binary',alpha=0.6)
        a[2].imshow(cleaned_img)
        a[3].imshow(fragments)
        # a[0,3].imshow(1-position,cmap='binary',alpha=0.6)
        # a[1,0].imshow(masked_img)
        # bool_mask = cnn_position_mask.astype(bool)
        # Create an overlay with green color where the mask is True
        # overlay = np.zeros_like(img_np)
        # overlay[bool_mask] = [0, 255, 0]  # Green color
        # Combine the original image and the overlay
        # alpha = 0.5  # Adjust alpha to control the transparency of the overlay
        # output = cv2.addWeighted(img_np, 1, overlay, alpha, 0)

        # a[1, 2].imshow(cnn_mask, cmap='binary')
        # a[1,3].imshow(single_cnn_output)
        #
        # a[1, 0].imshow(cnn_position_mask, cmap='binary')
        # a[1, 1].imshow(cnn_position_output)

        plt.show()

    # save_rgb_image(img_np,'./temp_result/method/cytassist/'+str(i)+'_with_fiducial.png')
    # save_rgb_image(single_cnn_output, './temp_result/method/cytassist/' + str(i) + '_without_fiducial.png')
    # save_rgb_image(img_tissue, './temp_result/method/cytassist/' + str(i) + '_tissue.png')


print('cnn iou:')
print(fiducial_ious_cnn/cnt)
print('number of samples:')
print(cnt)
# print('cnn_position iou:')
# print(fiducial_ious_cnn_position/cnt)
# print('binary iou:')
# print(binary_ious/cnt)
#
# print('current data set done')

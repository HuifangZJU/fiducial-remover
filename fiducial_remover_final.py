import argparse
import statistics
import time

from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
from cnn_io import *
from hough_utils import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import fiducial_utils
from multiprocessing import Pool, cpu_count
import scipy.ndimage as ndi
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor

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

def get_image_var(img_pil):
    h, w = img_pil.size
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new), Image.ANTIALIAS)
    img_np = np.array(img_pil)
    img_var = transforms_rgb(img_pil)
    img_var = torch.unsqueeze(img_var, dim=0).to(device)
    return img_var,img_np

def get_cnn_mask(img_var):
    cnn_mask_var = circle_generator(img_var)
    cnn_mask = cnn_mask_var.cpu().detach().numpy().squeeze()
    cnn_mask = np.transpose(cnn_mask, (0, 1))
    cnn_mask = normalize_array(cnn_mask)
    return cnn_mask

def get_position_mask(img_var, use_gaussian=True):
    position_var = position_generator(img_var)
    position = position_var.cpu().detach().numpy().squeeze()
    position = np.transpose(position, (0, 1))

    if use_gaussian:
        position = apply_gaussian_kernel(position, sigma=1.5)
    _, _, w, h = img_var.shape
    position = get_image_mask_from_annotation([w, h], position, patch_size)
    position = normalize_array(position)
    return position

def get_cnn_position_mask(cnn_mask,position):

    cnn_position_mask = cnn_mask*position

    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(cnn_position_mask, (3, 3), 0)

    threshold = 0.1  # This can be adjusted based on your observations
    _, binary_mask = cv2.threshold(blurred_mask, threshold, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)



    # Perform Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel,iterations=10)

    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Perform the morphological opening
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    return opened_mask





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
def run_geometric(img,run_square=True,return_radius=False):
    circles = run_circle_threhold(img, 9, circle_threshold=30, step=5)
    re_r = statistics.mode(circles[:,2])
    circles = run_circle_threhold(img, re_r, circle_threshold=int(2.2*re_r), step=1)
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
    array = array.astype('uint8')
    cv2.imwrite(filename, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR format

# For arrays with shape [m, n]
def save_gray_image(array, filename):
    # The input is expected to be in the range [0, 1], so we scale it to [0, 255]
    array = (array * 255).astype('uint8')
    cv2.imwrite(filename, array)


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
# Initialize generator
circle_generator = get_circle_Generator()
circle_generator.to(device)

position_generator = get_position_Generator()
position_generator.to(device)


# ------ main process -------
# manage input
patch_size = 32
transforms_rgb = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


test_image_path = '/home/huifang/workspace/data/imagelists/st_image_trainable_temp_fiducial.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
num_files = len(files)
for i in range(num_files):
    i=109
    print(str(num_files)+'---'+str(i))
    start_time = time.time()
    image_name = files[i]
    img_pil= Image.open(image_name.split(' ')[0])
    level = int(image_name.split(' ')[1])
    # if level !=2:
    #     continue
    img_var,img_np = get_image_var(img_pil)
    cnn_mask = get_cnn_mask(img_var)
    position = get_position_mask(img_var,use_gaussian=True)
    binary_position = binarize_array(position,0.5)

    # position_square_params = fit_square_to_mask(position)
    # cx, cy, s = position_square_params
    # s = abs(s)

    cnn_position_mask = get_cnn_position_mask(cnn_mask,position)
    cnn_mask_circles, cnn_mask_circle_figure, radius = run_geometric(cnn_mask * position, run_square=False,
                                                                      return_radius=True)
    end_time = time.time()
    # print(end_time-start_time)
    # continue
    # cnn_mask binary_position cnn_mask_circle_figure cnn_position_mask
    # save_rgb_image(img_np,'./test_fiducial_images/'+str(i)+'.png')
    #
    # save_gray_image(cnn_mask,'./test_fiducial_images/'+str(i)+'_mask1.png')
    # save_gray_image(binary_position, './test_fiducial_images/' + str(i) + '_mask2.png')
    # save_gray_image(cnn_mask_circle_figure, './test_fiducial_images/' + str(i) + '_mask3.png')
    # save_gray_image(cnn_position_mask, './test_fiducial_images/' + str(i) + '_mask4.png')
    # print('done')
    # continue



    # recs = process_and_visualize_mask(cnn_position_mask)

    # w = img_np.shape[0]
    # h = img_np.shape[1]
    # half_w = int(0.5*w)
    # half_h = int(0.5 * h)
    # recs = [[0,0,half_w,half_h],[0,half_h,half_w,h],[half_w,0,w,half_h],[half_w,half_h,w,h]]
    # recs = []
    # patch_size = 16
    # for circle in cnn_mask_circles:
    #     recs.append([circle[1]-patch_size,circle[0]-patch_size,circle[1]+patch_size,circle[0]+patch_size])
    #
    #
    # imgout = img_np.copy()
    # cnn_position_mask_var = torch.from_numpy(1-cnn_position_mask).type(Tensor)
    # cnn_position_mask_var = cnn_position_mask_var.unsqueeze(0).unsqueeze(0)
    # start_time = time.time()
    #
    # for rec in recs:
    #
    #     delta_x = find_nearest_multiple_of_32(rec[2] - rec[0])
    #     delta_y = find_nearest_multiple_of_32(rec[3] - rec[1])
    #     rec_img_np = cnn_position_mask[rec[0]:rec[0]+delta_x,rec[1]:rec[1]+delta_y]
    #     xidx,yidx = np.where(rec_img_np == 1.0)
    #     orig_xidx = xidx+ rec[0]
    #     orig_yidx = yidx + rec[1]
    #     recover_var = getReconstructedImg(img_var[:,:,rec[0]:rec[0]+delta_x,rec[1]:rec[1]+delta_y], cnn_position_mask_var[:,:,rec[0]:rec[0]+delta_x,rec[1]:rec[1]+delta_y], device,
    #                                       Tensor, 200)
    #     recover_np = torch_to_np(recover_var).squeeze()
    #     recover_np = np.transpose(recover_np, (1, 2, 0))
    #     imgout[orig_xidx,orig_yidx,:] = recover_np[xidx,yidx,:]*255
    #     plt.imshow(imgout)
    #     plt.show()
    # end_time = time.time()
    # print(end_time-start_time)


    # masked_img =img_np*np.expand_dims(position, axis=-1)
    # masked_img = masked_img.astype(np.uint8)
    # cnn_mask_circles,cnn_mask_circles_figure,radius = run_geometric(cnn_mask*position,run_square=False,return_radius=True)
    # masked_img_circles,masked_img_circles_figure,masked_img_central_square = run_geometric(masked_img)






    f,a = plt.subplots(2,3,figsize=(15, 10))
    a[0,0].imshow(img_np)
    a[0,1].imshow(1-cnn_mask,cmap='gray')
    # a[0,1].imshow(1-cnn_mask,cmap='gray')
    a[0,2].imshow(1-position,cmap='gray')
    #
    # square = np.array([
    #     [cx - s / 2, cy - s / 2],
    #     [cx + s / 2, cy - s / 2],
    #     [cx + s / 2, cy + s / 2],
    #     [cx - s / 2, cy + s / 2]
    # ])
    # a[0, 2].plot([square[0][1], square[1][1], square[2][1], square[3][1], square[0][1]],[square[0][0], square[1][0], square[2][0], square[3][0], square[0][0]], 'r-')
    # a[1,0].imshow(masked_img)
    bool_mask = cnn_position_mask.astype(bool)
    # Create an overlay with green color where the mask is True
    overlay = np.zeros_like(img_np)
    overlay[bool_mask] = [0, 255, 0]  # Green color

    # Combine the original image and the overlay
    alpha = 0.5  # Adjust alpha to control the transparency of the overlay
    output = cv2.addWeighted(img_np, 1, overlay, alpha, 0)
    a[1,0].imshow(output)
    a[1,1].imshow(1-cnn_mask_circle_figure,cmap='gray')
    a[1,2].imshow(1-cnn_position_mask,cmap='gray')
    # a[1,1].plot(cnn_central_square[[0, 1, 2, 3, 0], 0], cnn_central_square[[0, 1, 2, 3, 0], 1], c='red')

    # a[1,2].imshow(masked_img_circles_figure)
    # a[1, 2].plot(masked_img_central_square[[0, 1, 2, 3, 0], 0], masked_img_central_square[[0, 1, 2, 3, 0], 1], c='red')
    plt.show()



    # ------from model------#
    # mask_var = generator(img_var)

    # img_np = img_np.astype(np.uint8)
    # mask_np = mask_np.astype(np.uint8)
    # t1 = time.time()
    # recovered_image = run(img_np, mask_np, circles)
    # t2 = time.time()
    # print('Recover cost %.2f seconds.' % (t2-t1))
    # save(recovered_image,root_path+dataset_name+'/'+ image_name+'/masks/'+ masktype + '_result.png')
    # print('current image done')
    # # plt.imshow(recovered_image)
    # # plt.show()
print('current data set done')

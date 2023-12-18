import argparse
import statistics

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import yaml
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
from cnn_io import *
from hough_utils import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import fiducial_utils





def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def get_masked_image(image,circle):
    for c in circle:
        cv2.circle(image, (c[0], c[1]), c[2], 1,-1)
    return image



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


def get_inpainting_result(mask):
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

def run_geometric(img,position,return_radius=False):
    circles = run_circle_threhold(img, 10, circle_threshold=30, step=5)
    re_r = statistics.mode(circles[:,2])
    circles = run_circle_threhold(img, re_r, circle_threshold=int(2.2*re_r), step=1)
    new_circles=[]
    for x,y,r in circles:
        if position[y,x]>0.1:
            new_circles.append([x,y,r+3])
    circles = np.asarray(new_circles)
    hough_mask = np.zeros(img.shape[:2])
    hough_mask = get_masked_image(hough_mask, circles)
    if return_radius:
        return circles, hough_mask,re_r
    else:
        return circles, hough_mask


def transform_points(points, tx, ty, scale_x, scale_y, angle):
    scaling_matrix = np.array([[scale_x, 0], [0, scale_y]])
    rotation_matrix = R.from_euler('z', angle, degrees=True).as_matrix()[:2, :2]
    translation_vector = np.array([tx, ty])

    transformed_points = np.dot(points, scaling_matrix)
    transformed_points = np.dot(transformed_points, rotation_matrix) + translation_vector
    return transformed_points

def split_array(arr, cond):
  return arr[cond], arr[~cond]


def save_rgb_image(array, filename):
    # The input is expected to be in the range [0, 1], so we scale it to [0, 255]
    array = array.astype('uint8')
    cv2.imwrite(filename, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR format

# For arrays with shape [m, n]
def save_gray_image(array, filename):
    # The input is expected to be in the range [0, 1], so we scale it to [0, 255]
    array = (array * 255).astype('uint8')
    cv2.imwrite(filename, array)

def get_imgnp(image_path):
    img_pil = Image.open(image_path)
    h, w = img_pil.size
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new), Image.ANTIALIAS)
    img_np = np.array(img_pil)
    return img_np

# ------ device handling -------
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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

test_image_path = '/home/huifang/workspace/data/imagelists/st_image_with_aligned_fiducial.txt'
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
num_files = len(files)
for i in range(0,num_files):
    print(str(num_files)+'---'+str(i))
    start_time = time.time()
    image_name = files[i].split(' ')
    image_path = image_name[0]
    aligned_path = image_name[1].rstrip('\n')

    img_pil = Image.open(image_path)
    h, w = img_pil.size
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new), Image.ANTIALIAS)
    img_np = np.array(img_pil)

    h_ratio = h_new/h
    w_ratio = w_new/w
    aligned_image = plt.imread(aligned_path)
    transposed_fiducial, scale = fiducial_utils.runCircle(aligned_path)
    # print(transposed_fiducial)
    transposed_fiducial[:,0] = transposed_fiducial[:,0]*h_ratio
    transposed_fiducial[:, 1] = transposed_fiducial[:, 1] * w_ratio

    mask = generate_mask(img_np.shape[:2],transposed_fiducial,-1)
    mask = mask.astype(np.uint8)


    inpainter_image = np.transpose(img_np,(2,0,1))
    inpainter_image = inpainter_image.astype('float32')/255
    output = get_inpainting_result(mask)

    end_time = time.time()
    print(end_time-start_time)
    # single_cnn_output = get_inpainting_result(single_cnn_mask)
    # cnn_output = get_inpainting_result(cnn_mask)
    # save_rgb_image(direct_output,'./temp_result/circle/'+str(i)+'.png')
    # save_rgb_image(cnn_position_output, './temp_result/network/' + str(i) + '.png')
    # continue


    f,a = plt.subplots(2,2,figsize=(15, 20))
    a[0,0].imshow(img_np)
    a[0,1].imshow(aligned_image)
    a[1,0].imshow(img_np)
    a[1,0].imshow(1-mask,cmap='binary',alpha=0.4)
    a[1,1].imshow(output)
    plt.show()


print('current data set done')

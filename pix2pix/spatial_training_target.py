import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm

base_value = 32

def find_nearest_multiple_of_base(x):
    base = base_value
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)

def binarize_mask(mask_file, threshold=128):
    """
    Convert a grayscale mask to a binary mask.
    :param mask_file: Path to the mask image file.
    :param threshold: Threshold value to binarize the mask. Defaults to 128.
    :return: Binarized mask as a numpy array.
    """
    # Load the mask image
    mask = Image.open(mask_file).convert('L')  # Convert to grayscale
    h, w = mask.size
    h_new = find_nearest_multiple_of_base(h)
    w_new = find_nearest_multiple_of_base(w)
    mask = mask.resize((h_new, w_new), Image.ANTIALIAS)
    # Convert to numpy array and apply threshold
    mask_array = np.array(mask)
    binary_mask = (mask_array > threshold).astype(np.uint8)
    return binary_mask

def generate_patch_patterns(mask_file, patch_size=base_value):
    mask = binarize_mask(mask_file)
    mask_tensor = torch.tensor(mask).unsqueeze(0)

    # Initialize accumulators for positive and negative patterns
    positive_accumulator = torch.zeros((1, patch_size, patch_size))
    positive_count, negative_count = 0, 0

    # Divide the mask into patches and accumulate
    for i in range(0, mask_tensor.shape[1], patch_size):
        for j in range(0, mask_tensor.shape[2], patch_size):
            patch = mask_tensor[:, i:i+patch_size, j:j+patch_size]
            if patch.sum() > 0:  # Positive patch (contains ones)
                positive_accumulator += patch
                positive_count += 1

    # Calculate average patterns
    positive_pattern = positive_accumulator / positive_count
    positive_mean = positive_pattern.mean()
    # print(f"Mean value of Positive Pattern: {positive_mean.item()}")
    return positive_mean

    # Save patterns
    # torch.save(positive_pattern, save_path_positive)
    # torch.save(negative_pattern, save_path_negative)

    # Visualize patterns
    # plt.subplot(1, 2, 1)
    # plt.imshow(positive_pattern.squeeze(), cmap='gray')
    # plt.title('Average Positive Pattern')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(negative_pattern.squeeze(), cmap='gray')
    # plt.title('Average Negative Pattern')
    # plt.axis('off')
    #
    # plt.show()

# Example usage

image_list = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
# image_list = '/home/huifang/workspace/data/imagelists/st_auto_trainable_images.txt'
f = open(image_list, 'r')
fiducial_images = f.readlines()
positive_means=[]
for i in range(0,len(fiducial_images)):

    image_name = fiducial_images[i].split(' ')[0]
    mask_file = image_name.split('.')[0] + '_ground_truth.png'
    positive_mean = generate_patch_patterns(mask_file)
    positive_means.append(positive_mean)

mu, std = norm.fit(positive_means)

# Plot the Gaussian function
plt.hist(positive_means, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = {:.2f},  std = {:.2f}".format(mu, std)
plt.title(title)
plt.show()
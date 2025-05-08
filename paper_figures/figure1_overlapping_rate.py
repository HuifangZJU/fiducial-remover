import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import ListedColormap
from skimage import measure
# ------ device handling -------
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'

def binarize_array(array, threshold):
    """
    Binarizes a numpy array based on a threshold determined by the given percentile.

    :param array: numpy array to be binarized
    :param percentile: percentile value used to determine the threshold, defaults to 50 (median)
    :return: binarized numpy array
    """
    binary_array = (array >= threshold).astype(int)

    return binary_array


def calculate_iou(mask1, mask2):
    # Ensure the masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0.0
    return iou




def save_result(images,filename_prefix):

    for i, img_array in enumerate(images):
        # Check if the image is grayscale, RGB, or RGBA
        if img_array.ndim == 2:  # Grayscale
            img = Image.fromarray(img_array)
        elif img_array.ndim == 3:
            if img_array.shape[2] == 3:  # RGB
                img = Image.fromarray(img_array)
            elif img_array.shape[2] == 4:  # RGBA
                img = Image.fromarray(img_array, 'RGBA')
            else:
                raise ValueError(f"Image at index {i} has an unsupported channel size: {img_array.shape[2]}")
        else:
            raise ValueError(f"Image at index {i} has an unsupported shape: {img_array.shape}")

        # Define the output file name with the prefix
        output_filename = f"{filename_prefix}_image_{i}.png"

        # Save the image
        img.save(output_filename)
        print(f"Saved: {output_filename}")


path = 'test.txt'
f = open(path, 'r')
lines = f.readlines()
f.close()
percentages = []


# Loop through each line, calculate the percentage, and store it in the list
for i in range(len(lines)):
    line = lines[i]
    line = line.split(' ')
    gt_file = line[0][:-4] + '.npy'
    gt = np.load(gt_file)

    # Select the fourth column
    fourth_column = gt[:, 3]

    # Replace 2s with 1s
    fourth_column = np.where(fourth_column > 1, 1, fourth_column)

    # Calculate the percentage of values greater than 0
    percentage_of_markers_overlapping_with_tissue = np.mean(fourth_column > 0) * 100
    percentages.append(percentage_of_markers_overlapping_with_tissue)

# Convert percentages to a numpy array for easier processing
percentages = np.array(percentages)
# np.save("overlapping_rate.npy",percentages)
# print("saved")
# test = input()

# Define the conditions for each range
counts = [
    np.sum(percentages == 0),                # No overlapping (0%)
    np.sum((percentages > 0) & (percentages < 10)),   # Slight overlapping (0%-10%)
    np.sum((percentages >= 10) & (percentages < 30)), # Moderate overlapping (10%-30%)
    np.sum((percentages >= 30) & (percentages < 50)), # High overlapping (30%-50%)
    np.sum(percentages >= 50)                # Very high overlapping (>50%)
]
labels = ["No overlap\n(0%)", "Slight\noverlap\n(0%-10%)",
          "Moderate\noverlap\n(10%-30%)", "High overlap\n(30%-50%)",
          "Very high\noverlap\n(50%-100%)"]

# Calculate the total to get percentages
total_count = sum(counts)
percentages = [count / total_count * 100 for count in counts]

# Use a subtle color palette from seaborn's pastel colors
# colors = sns.color_palette("RdBu", len(counts))

# Define angles for each segment
angles = np.linspace(0, 2 * np.pi, len(counts) + 1)

# Adjust radius slightly to have small differences between each segment
adjusted_counts = [radius * 0.1 + 10 for radius in counts]  # This adds subtle radius variation

# Plotting
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': 'polar'})
# colors = ['#eca8a9', '#74aed4', '#d3e2b7', '#cfafd4', '#f7c97e']
colors = ['#c6d182','#eae0e9', '#e0c7e3', '#ae98b6', '#846e89']
bars = ax.bar(
    angles[:-1],  # Starting angle of each bar
    adjusted_counts,  # Radius (length) of each bar with subtle differences
    width=angles[1] - angles[0],  # Width of each bar
    color=colors,
    edgecolor='white',
    linewidth=1.5,
    alpha=0.85
)

ax.text(
    angles[0], adjusted_counts[0]+1.5, labels[0], ha='center', va='center',
    fontsize=24, color='black'
)

ax.text(
    angles[1], adjusted_counts[1]+4.5, labels[1], ha='center', va='center',
    fontsize=24, color='black'
)

ax.text(
    angles[2]-0.2, adjusted_counts[2]+4.2, labels[2], ha='center', va='center',
    fontsize=24, color='black'
)

ax.text(
    angles[3], adjusted_counts[3]+3.8, labels[3], ha='center', va='center',
    fontsize=24, color='black'
)

ax.text(
    angles[4], adjusted_counts[4]+5.3, labels[4], ha='center', va='center',
    fontsize=24, color='black'
)



# Adding labels with custom alignment and position
for angle, radius, label, pct in zip(angles, adjusted_counts, labels, percentages):
    # Position text halfway out each segment for label

    # Add percentage at the center of each segment
    ax.text(
        angle, radius / 2, f"{pct:.1f}%", ha='center', va='center',
        fontsize=26, color='black', weight='bold'
    )

# Style adjustments
ax.set_yticklabels([])  # Hide radial labels
ax.set_xticks([])  # Hide angle labels
ax.spines['polar'].set_visible(False)  # Hide the circular frame
ax.grid(False)  # Turn off the grid
ax.set_theta_zero_location("N")  # Set the zero angle to the top
# plt.savefig('./figures/1.png', dpi=300)
# # Title
plt.title("Distribution of Markers Overlapping with Tissue", fontsize=16, weight='bold')
plt.show()

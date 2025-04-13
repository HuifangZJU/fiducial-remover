import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy as np
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
# # # Read the image file paths from the text file
# image_paths = []
# # levels = []
# with open('/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_trainable_fiducial.txt', 'r') as file:  # Replace 'images.txt' with your actual file path
#     lines  = file.readlines()
#     ids=[17,28,31,36,51,118,141,151,161,162,73,74,86,87,92,105,106,136,149,157,163,67,158,156,4]
#     ids.sort()
#     for id in ids:
#         line = lines[id]
#         # image_path = line.strip().split()[0]
#         image_path = '/home/huifang/workspace/code/fiducial_remover/location_annotation/'+str(id)+'.png'
#         level = int(line.strip().split()[1])
#         # if level==3:
#         #     continue
#         image_paths.append(image_path)
#         # levels.append(level)
#         # if len(image_paths) == 25:
#         #     break
# # # Define the grid size
# rows, cols = 5, 5  # Define the number of rows and columns for the grid
# new_size = (1024, 1024)  # New size for each image, adjust as needed
#
# # Create a new blank image with the correct size
# concatenated_image = Image.new('RGB', (new_size[0] * cols, new_size[1] * rows))
#
# # Load, resize, and paste each image into the blank canvas
# for idx, image_path in enumerate(image_paths):
#     img = Image.open(image_path)
#     img = img.resize(new_size)
#     row_idx, col_idx = divmod(idx, cols)
#     concatenated_image.paste(img, (col_idx * new_size[0], row_idx * new_size[1]))
#
# # Display the concatenated image
# plt.figure(figsize=(cols, rows))  # Figure size might need to be adjusted
# plt.imshow(concatenated_image, cmap='gray')
# plt.axis('off')  # Turn off the axis
# plt.show()

# for image_path,level in zip(image_paths,levels):
#     if level == 1:
#         continue
#     img = Image.open(image_path)
#     plt.figure(figsize=(15,15))
#     plt.imshow(img)
#     plt.show()


# test_image_path = '/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_trainable_fiducial.txt'
# result_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/circle/'
# # test_image_path = '/home/huifang/workspace/data/imagelists/st_cytassist.txt'
# #
# f = open(test_image_path, 'r')
# files = f.readlines()
# f.close()
# num_files = len(files)
# fiducial_ious=0
# average_ious=0
# cnt=0
# for i in range(num_files-1,0,-1):
#     print(str(num_files)+'---'+str(i))
#     image_name = files[i]
#     level = int(image_name.split(' ')[1])
#     if level == 1:
#         continue
#     image_name = image_name.split(' ')[0]
#
#     image = plt.imread(image_name)
#     brightness_factor = 1.2
#     image = np.clip(image * brightness_factor, 0, 1)
#     image_clean = plt.imread(result_path+str(i)+'.png')
#     brightness_factor = 1.2
#     image_clean = np.clip(image_clean * brightness_factor, 0, 1)
#
#     f,a = plt.subplots(1,2,figsize=(20,10))
#     a[0].imshow(image)
#     a[1].imshow(image_clean)
#     plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# methods = ['10x', 'Vispro']
# values = [2, 3]
# ground_truth = 3
# colors = ['royalblue', 'gold']
#
# # Positions for bars
# positions = [0.45, 0.52]  # Adjust the positions to control spacing
#
# # Creating the plot
# plt.figure(figsize=(4, 5))
#
# # Control bar location using positions and bar width using width parameter
# plt.bar(positions, values, yerr=[1, 0], color=colors, capsize=10, width=0.05)  # Adjust the width
#
# # Set the x-ticks to match the positions of bars
# plt.xticks(positions, methods)
#
# # Set x-axis limits to compress the space between bars if necessary
# plt.xlim(0.37, 0.6)  # This can be adjusted for more control
#
# # Add labels and title
# # plt.xlabel('Methods')
# plt.ylabel('Number of Tissue Segments')
# # plt.title('Comparison of Methods with Ground Truth')
# plt.savefig('/home/huifang/workspace/code/fiducial_remover/temp_result/application/figure/1.png', dpi=600)
# # Show plot
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['10x', 'Vispro']
values = [2, 3]
ground_truth = 3
colors = ['royalblue', 'gold']

# Calculate L1 distance
l1_distances = np.abs(np.array(values) - ground_truth)


# Define positions of the bars (manually setting x-values)
positions = [0.42, 0.58]  # Adjust these values to control bar spacing

# Define width of the bars
bar_width = 0.1 # Adjust the bar width

# Create the plot
plt.figure(figsize=(8, 8))

# Bar plot with specified positions and width
bars = plt.bar(positions, values, color=colors, width=bar_width)

# Plot the ground truth as a horizontal line
plt.axhline(y=ground_truth, color='gray', linestyle='--')

# Add deviation lines from the bars to the ground truth
for i, (bar, value) in enumerate(zip(bars, values)):
    plt.plot([bar.get_x() + bar.get_width()/2, bar.get_x() + bar.get_width()/2],
             [value, ground_truth], color='black', linestyle='-', linewidth=1.5)

# Add labels and title
# plt.xlabel('Methods')
# plt.ylim(0, 13)
plt.xlim(0.25, 0.75)  # This can be adjusted for more control
plt.ylabel('Number of Tissue Segments',fontsize=36, fontfamily='Arial')
# plt.title('L1 Distance (Deviation) from Ground Truth')

# Add a 'Ground Truth' label in the right upper corner
# plt.text(1.0, ground_truth + 0.1, 'Ground Truth', ha='right', va='bottom', color='gray')

ax = plt.gca()  # Get current axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set x-ticks to match the bar positions and customize font
# plt.yticks([0,2,4,6,8,10,12,13],fontsize=36)
plt.yticks([0,1,2,3],fontsize=36)
plt.xticks(positions, methods, fontsize=36, fontfamily='Arial')
plt.savefig('/home/huifang/workspace/code/fiducial_remover/temp_result/application/figure/1.png', dpi=600)
# Show the plot
# plt.show()

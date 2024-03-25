import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# # Read the image file paths from the text file
# image_paths = []
# levels = []
# with open('/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_trainable_fiducial.txt', 'r') as file:  # Replace 'images.txt' with your actual file path
#     for line in file:
#         image_path = line.strip().split()[0]
#         level = int(line.strip().split()[1])
#         # if level==3:
#         #     continue
#         image_paths.append(image_path)
#         levels.append(level)
#         if image_paths == 100:
#             break
# # # Define the grid size
# rows, cols = 10, 10  # Define the number of rows and columns for the grid
# new_size = (100, 100)  # New size for each image, adjust as needed
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
#
# # for image_path,level in zip(image_paths,levels):
# #     if level == 1:
# #         continue
# #     img = Image.open(image_path)
# #     plt.figure(figsize=(15,15))
# #     plt.imshow(img)
# #     plt.show()


test_image_path = '/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_trainable_fiducial.txt'
result_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/circle/'
# test_image_path = '/home/huifang/workspace/data/imagelists/st_cytassist.txt'
#
f = open(test_image_path, 'r')
files = f.readlines()
f.close()
num_files = len(files)
fiducial_ious=0
average_ious=0
cnt=0
for i in range(num_files-1,0,-1):
    print(str(num_files)+'---'+str(i))
    image_name = files[i]
    level = int(image_name.split(' ')[1])
    if level == 1:
        continue
    image_name = image_name.split(' ')[0]

    image = plt.imread(image_name)
    brightness_factor = 1.2
    image = np.clip(image * brightness_factor, 0, 1)
    image_clean = plt.imread(result_path+str(i)+'.png')
    brightness_factor = 1.2
    image_clean = np.clip(image_clean * brightness_factor, 0, 1)

    f,a = plt.subplots(1,2,figsize=(20,10))
    a[0].imshow(image)
    a[1].imshow(image_clean)
    plt.show()

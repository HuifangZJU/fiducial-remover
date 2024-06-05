import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def save_image(array,filename,format="RGB"):
    if array.max()<1.1:
        array = 255 * array
    array = array.astype(np.uint8)
    array = Image.fromarray(array)
    if format == "RGB":
        array = array.convert('RGB')
    else:
        assert(format == "L")
        array = array.convert('L')
    array.save(filename)

def annotate_patches(image_size, patch_size, circles):
    w,h = image_size
    num_patches_w = w // patch_size
    num_patches_h = h // patch_size

    annotation = np.zeros((num_patches_w, num_patches_h), dtype=int)
    for i in range(num_patches_w):
        for j in range(num_patches_h):
            patch_x = i * patch_size
            patch_y = j * patch_size
            patch_rect = (patch_x, patch_y, patch_size, patch_size)

            for circle in circles:
                circle_x, circle_y, circle_radius = circle[:3]
                circle_radius = circle_radius+1
                circle_rect = (circle_y - circle_radius,circle_x - circle_radius,  2 * circle_radius, 2 * circle_radius)


                if rectangles_intersect(patch_rect, circle_rect):
                    annotation[i, j] = 1
                    # plt.imshow(image[patch_x:patch_x + step, patch_y:patch_y + step])
                    # print(annotation[i, j])
                    # plt.show()
                    break
    return annotation

def calculate_centers_and_scales_from_outlines(outlines):
    properties = []

    for outline in outlines:
        # Assuming outline is a flattened array with even indices for x and odd indices for y
        xs = outline[:, 0, 0]
        ys = outline[:, 0, 1]

        # Calculate center
        center_x = np.mean(xs)
        center_y = np.mean(ys)

        # Calculate scale (width and height)
        width = np.max(xs) - np.min(xs)
        height = np.max(ys) - np.min(ys)
        scale = 0.5*np.max([width,height])

        properties.append([center_x, center_y, scale])

    return np.asarray(properties,dtype=int)

# Set the directory paths
outlines_dir = '/home/huifang/workspace/code/fiducial_remover/cellpose_results/txt_outlines/'
images_dir = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
f = open(images_dir, 'r')
files = f.readlines()
f.close()
num_files = len(files)
# Plotting outlines and corresponding masks
for i in range(num_files):
    # Read the outline .txt file
    outline_path = os.path.join(outlines_dir, f'{i}_cp_outlines.txt')
    with open(outline_path, 'r') as f:
        outlines = f.readlines()
    outlines = [np.array(list(map(int, line.strip().split(',')))).reshape((-1, 1, 2)) for line in outlines]

    # Read the corresponding image file
    image_name = files[i]
    image_name = image_name.split(' ')[0]
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

    # Create an empty array for the binary mask
    mask = np.zeros_like(img)

    # Draw the filled polygons (outlines) on the mask
    cv2.fillPoly(mask, outlines, color=(1))

    fake_circles = calculate_centers_and_scales_from_outlines(outlines)

    save_image(mask, image_name.split('.')[0] + '_cellpose.png', format="L")
    np.save(image_name.split('.')[0] + '_cellpose.npy', fake_circles)
    continue


    fake_binary_mask = annotate_patches(img.shape[:2], 32,fake_circles)
    # Display the original image with outlines
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    for outline in outlines:
        plt.plot(outline[:, 0, 0], outline[:, 0, 1], 'r-')  # Plot the outlines on the image
    plt.title('Original Image with Outlines')
    plt.axis('off')

    # Display the binary mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title('Binary Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(fake_binary_mask)
    plt.title('Binary Mask -- position')
    plt.axis('off')

    plt.show()


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io, measure, morphology, color, util


def process_image_for_holes(image):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Closing operation
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Removing small objects and filling small holes
    cleaned = morphology.remove_small_objects(closing > 0, min_size=200)
    holes = morphology.remove_small_holes(cleaned, area_threshold=200)

    return holes

def add_contours(image, holes_image):
    # Copy the original image to not modify it
    image_with_contours = image.copy()

    # Find contours from the hole image
    contours, _ = cv2.findContours(util.img_as_ubyte(holes_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    return image_with_contours

def detect_keypoints(image):
    """Detects keypoints in an H&E stained image using ORB."""
    orb = cv2.ORB_create()
    keypoints = orb.detect(image, None)
    keypoints, descriptors = orb.compute(image, keypoints)
    return keypoints, descriptors


def display_keypoints(image1, keypoints1, image2, keypoints2):
    """Displays images with keypoints side by side."""
    image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=0)
    image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0), flags=0)

    # Convert BGR to RGB for matplotlib display
    # image1_with_keypoints = cv2.cvtColor(image1_with_keypoints, cv2.COLOR_BGR2RGB)
    # image2_with_keypoints = cv2.cvtColor(image2_with_keypoints, cv2.COLOR_BGR2RGB)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0,0].imshow(image1)
    axes[0,0].set_title('Image 1')
    axes[0,0].axis('off')

    axes[0,1].imshow(image2)
    axes[0,1].set_title('Image 2')
    axes[0,1].axis('off')

    axes[1,0].imshow(image1_with_keypoints)
    axes[1,0].set_title('Image 1 with Keypoints')
    axes[1,0].axis('off')

    axes[1,1].imshow(image2_with_keypoints)
    axes[1,1].set_title('Image 2 with Keypoints')
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show()


path1 = '/home/huifang/workspace/code/fiducial_remover/temp_result/overlap_patch/with_fiducial/'
path2 = '/home/huifang/workspace/code/fiducial_remover/temp_result/overlap_patch/without_fiducial/'
contents = os.listdir(path1)

# Print the list of contents
for i in range(44,len(contents),2):
    item = contents[i]
    print(i)
    image1 = io.imread(path1+item)
    image2 = io.imread(path2+item)
    # Process images for hole detection
    holes_image1 = process_image_for_holes(image1)
    holes_image2 = process_image_for_holes(image2)

    # Add contours to the original images
    image1_with_contours = add_contours(image1, holes_image1)
    image2_with_contours = add_contours(image2, holes_image2)

    # Displaying the results
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    ax[0, 0].imshow(image1)
    ax[0, 0].set_title("Original Image 1")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(image1_with_contours)
    ax[0, 1].set_title("Image 1 with Contours")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(image2)
    ax[1, 0].set_title("Original Image 2")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(image2_with_contours)
    ax[1, 1].set_title("Image 2 with Contours")
    ax[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    #
    # # Extract morphological features
    # features_image1 = extract_morphological_features(image1)
    # features_image2 = extract_morphological_features(image2)
    #
    #
    # # Compare morphological features (basic comparison)
    # # You can extend this part for more in-depth analysis
    # print("Number of features in image1:", len(features_image1))
    # print("Number of features in image2:", len(features_image2))
    #
    # # Detect and display keypoints
    # keypoints_image1, _ = detect_keypoints(image1)
    # keypoints_image2, _ = detect_keypoints(image2)
    #
    # print("Number of keypoints in image1:", len(keypoints_image1))
    # print("Number of keypoints in image2:", len(keypoints_image2))
    #
    # # Display images with keypoints
    # display_keypoints(image1, keypoints_image1, image2, keypoints_image2)

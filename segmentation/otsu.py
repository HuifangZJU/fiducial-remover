import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Read the image (OpenCV loads BGR by default)
for i in range(25,167):
    print(i)

    # original_bgr = cv2.imread("/home/huifang/workspace/code/fiducial_remover/location_annotation/"+str(i)+".png")
    original_bgr = cv2.imread("/media/huifang/data/fiducial/tiff_data/151508/spatial/tissue_hires_image_0.png")

    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)


    # 2) Apply a blur to reduce noise (optional but recommended)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)



    # 3) Otsu's thresholding to get a binary mask
    _, binary_mask = cv2.threshold(
        gray_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Create a copy of the original
    segmentation_result = original_bgr.copy()

    # Replace background pixels (where mask == 0) with white (BGR = 255, 255, 255)
    segmentation_result[binary_mask == 255] = (255, 255, 255)

    # Now save the result to a file
    cv2.imwrite("/media/huifang/data/fiducial/tiff_data/151508/spatial/otsu.png", segmentation_result)


    # 'binary_mask' is a 2D array with 0 for background, 255 for foreground

    # 4) Create a 3-channel color mask the same size as original
    #    We'll tint the foreground in green (BGR = (0, 255, 0))
    overlay_color = np.zeros_like(original_bgr, dtype=np.uint8)
    overlay_color[binary_mask == 0] = (0, 255, 0)   # Foreground in bright green

    # 5) Alpha blend the overlay with original
    #    alpha=0.3 means the overlay is 30% visible, original is 70%
    alpha = 0.3
    blended = cv2.addWeighted(original_bgr, 1 - alpha, overlay_color, alpha, 0)

    # 6) For plotting in matplotlib, convert BGR -> RGB
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    # 7) Show side by side
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(blended_rgb)
    plt.title("Otsu Foreground Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

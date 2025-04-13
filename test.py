import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import Image, ImageEnhance


def adjust_brightness_contrast(image: np.ndarray, brightness_factor: float, contrast_factor: float) -> np.ndarray:
    """
    Adjust brightness and contrast of an image.

    Parameters:
    - image: Input image as a NumPy ndarray.
    - brightness_factor: Factor to scale the brightness (e.g., 0.6 for 60% brightness).
    - contrast_factor: Factor to scale the contrast (e.g., 0.6 for 60% contrast).

    Returns:
    - Adjusted image as a NumPy ndarray.
    """
    # Normalize the image to [0, 1]
    normalized_image = image / 255.0

    # Adjust contrast: scale pixel values around the mean
    mean_pixel_value = normalized_image.mean()
    contrast_adjusted = (normalized_image - mean_pixel_value) * contrast_factor + mean_pixel_value

    # Adjust brightness: scale all pixel values
    brightness_adjusted = contrast_adjusted * brightness_factor

    # Clip values to [0, 1] and convert back to [0, 255]
    adjusted_image = np.clip(brightness_adjusted, 0, 1) * 255
    return adjusted_image.astype(np.uint8)

def visualize_and_save_part_of_image(image_path, crop_box, output_path):
    """
    Visualizes a part of the image and saves it to a PNG file.

    Parameters:
        image_path (str): Path to the input image.
        crop_box (tuple): Coordinates (left, upper, right, lower) to define the region to crop.
        output_path (str): Path to save the cropped image as a PNG.
    """
    # Load the image
    image = Image.open(image_path)



    # # # Crop the image
    image = image.crop(crop_box)

    # #
    # # Calculate the new dimensions
    new_width = image.width // 1
    new_height = image.height// 1
    #
    # # Resize the image
    cropped_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    #
    # Convert the image to a NumPy array
    image_array = np.array(cropped_image)
    # cropped_image = adjust_brightness_contrast(image_array, 0.5, 0.8)
    # cropped_image = Image.fromarray(cropped_image)

    # Extract the first channel (e.g., Red in RGBA or Cyan in CMYK)
    first_channel = image_array[..., 0]  # Assuming the first channel is at index 0

    # Convert the first channel back to an image
    cropped_image = Image.fromarray(first_channel.astype(np.uint8))
    # Create an ImageEnhance object for contrast adjustment
    enhancer = ImageEnhance.Contrast(cropped_image)

    # Increase the contrast (factor > 1.0 increases contrast)
    # contrast_factor = 1.9  # Adjust this value (e.g., 1.5 for 50% increase)
    contrast_factor = 2.5  # Adjust this value (e.g., 1.5 for 50% increase)
    cropped_image = enhancer.enhance(contrast_factor)


    # #
    # # # # Display the cropped part
    plt.figure(figsize=(6, 6))
    plt.imshow(cropped_image,cmap='gray')
    plt.axis('off')  # Hide axes
    plt.show()

    # Save the cropped part as a PNG
    cropped_image.save(output_path, format="PNG")
    print(f"Cropped part saved to {output_path}")

# Example usage
image_path = "/home/huifang/workspace/paper/fiducial application/final_version/replace figures/registration/original/cytassist_0_blended_image1.png"  # Path to your input image
output_path = image_path[:-4]+"_crop.png"  # Path to save the cropped image
crop_box = (1750, 123, 1999, 392)  # Define the region: (left, upper, right, lower)

visualize_and_save_part_of_image(image_path, crop_box, output_path)

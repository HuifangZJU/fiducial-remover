import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
registration_pair=[[1, 55],
[14, 21],
[14, 58],
[21, 58],
[22, 24],
[44, 98],
[48, 92],
[67, 100],
[69, 94],
[70, 127],
[71, 129],
[75, 121],
[79, 133],
[84, 102],
[90, 138],
[104, 139],
[105, 124],
[110, 163],
[114, 119],
[126, 128]]


def get_simpITK_registration_result(fixed_path, moving_path):
    fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    # fixed_image = sitk.Shrink(fixed_image, [4, 4])
    # moving_image = sitk.Shrink(moving_image, [4, 4])
    # Set up the B-spline transform
    transform = sitk.BSplineTransformInitializer(fixed_image, [3, 3], order=3)  # Control points grid size

    # Registration setup
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransform(transform)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-3, numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Perform registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Resample the moving image using the final transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered_image = resampler.Execute(moving_image)

    # Convert images to numpy arrays for visualization
    fixed_array = sitk.GetArrayViewFromImage(fixed_image)
    moving_array = sitk.GetArrayViewFromImage(moving_image)
    registered_array = sitk.GetArrayViewFromImage(registered_image)

    fixed_norm = normalize(fixed_array)
    moving_norm = normalize(moving_array)
    registered_norm = normalize(registered_array)
    return fixed_norm,moving_norm,registered_norm




# Normalize images to [0, 1] for overlay
def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


# # Create elegant overlays (Cyan for Moving, Magenta for Fixed)
# def create_overlay(fixed, moving):
#     # overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
#     # overlay[..., 0] = fixed  # Magenta - Red Channel
#     # overlay[..., 1] = moving  # Cyan - Green and Blue Channel
#     # overlay[..., 2] = moving
#     overlay = np.zeros((fixed.shape[0], fixed.shape[1]))
#     overlay = 0.5*fixed + 0.5*moving
#     return overlay

def create_overlay(fixed, moving, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Blend two 2D images (fixed & moving) in grayscale and apply CLAHE for contrast.

    Parameters:
      fixed, moving : 2D NumPy arrays of the same shape (float [0,1] or uint8 [0,255])
      clip_limit    : CLAHE clip limit (higher = more contrast)
      tile_grid_size: CLAHE tile grid size

    Returns:
      overlay : 2D float array in [0,1], ready for plt.imshow(cmap='gray')
    """
    # —1— Convert to float32 [0,1]
    f = fixed.astype(np.float32)
    m = moving.astype(np.float32)
    if f.max() > 1.0: f /= 255.0
    if m.max() > 1.0: m /= 255.0

    # —2— Normalize each image independently to [0,1]
    def norm01(img):
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn) if mx > mn else img

    f_n = norm01(f)
    m_n = norm01(m)

    # —3— Average them for a basic overlay
    combined = (f_n + m_n) / 2.0

    # —4— Apply CLAHE via OpenCV
    combined_uint8 = np.uint8(np.clip(combined * 255, 0, 255))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_uint8 = clahe.apply(combined_uint8)

    # —5— Convert back to float [0,1]
    overlay = enhanced_uint8.astype(np.float32) / 255.0
    return overlay

def save_image(image,path):
    overlay_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(overlay_uint8, mode="L")  # "L" = 8‑bit grayscale
    img.save(path)
    return img
# Example usage
# with_fiducial_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/cytassist/with_fiducial/'
# without_fiducial_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/cytassist/without_fiducial/'

with_fiducial_path='/media/huifang/data/fiducial/temp_result/application/registration/with_fiducial/'
without_fiducial_path ='/media/huifang/data/fiducial/temp_result/application/registration/without_fiducial/'

metric_with_fiducial=0
metric_without_fiducial=0
for i in range(0,len(registration_pair)):
# [79, 133],#
# [90, 138],
# for i in [12,14]:
    print(i)
    image_id = registration_pair[i]
    id1 = image_id[0]
    id2 = image_id[1]


    fixed_path = without_fiducial_path+str(i)+"/"+str(id1)+'.png'
    moving_path = without_fiducial_path+str(i)+"/"+str(id2)+'.png'

    # fixed_path = "/media/huifang/data/registration/humanpilot/151671/spatial/tissue_hires_image_0.png"
    # moving_path = "/media/huifang/data/registration/humanpilot/151672/spatial/tissue_hires_image_0.png"


    fixed_norm,moving_norm,registered_norm = get_simpITK_registration_result(fixed_path,moving_path)

    save_image(fixed_norm,"/media/huifang/data/fiducial/temp_result/vispro/registration/marker_free/"+str(i)+"_fixed.png")
    save_image(moving_norm,
                 "/media/huifang/data/fiducial/temp_result/vispro/registration/marker_free/" + str(i) + "_moving.png")
    save_image(registered_norm,
                     "/media/huifang/data/fiducial/temp_result/vispro/registration/marker_free/" + str(i) + "_registered.png")
    # test = input()


    # overlay_before = create_overlay(fixed_norm, moving_norm)
    # overlay_after = create_overlay(fixed_norm, registered_norm)
    # save_image(overlay_before,"/media/huifang/data/fiducial/temp_result/vispro/registration/"+str(i)+"_vispro_original.png")
    # save_image(overlay_after,
    #              "/media/huifang/data/fiducial/temp_result/vispro/registration/" + str(i) + "_vispro_registered.png")
    # test = input()




    # # Plot results
    # fig, axes = plt.subplots(1, 4, figsize=(18, 8))
    #
    # axes[0].imshow(fixed_norm, cmap='gray')
    # axes[0].set_title("Fixed Image")
    #
    # axes[1].imshow(moving_norm, cmap='gray')
    # axes[1].set_title("Moving Image")
    #
    # axes[2].imshow(overlay_before, cmap='gray')
    # axes[2].set_title("Overlay Before Registration (Magenta/Cyan)")
    #
    # axes[3].imshow(overlay_after, cmap='gray')
    # axes[3].set_title("Overlay After Registration (Magenta/Cyan)")
    #
    # for ax in axes:
    #     ax.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

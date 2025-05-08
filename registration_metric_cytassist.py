import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

from sklearn.metrics import mutual_info_score
from skimage.transform import resize


# def calculate_ssim(image1, image2, visualization=False):
#     # Resize image2 if needed
#     if image1.shape != image2.shape:
#         image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
#
#     # Create a mask where both images are non-zero
#     mask = (image1 > 0) & (image2 > 0)
#
#     if np.sum(mask) == 0:
#         return 0  # Avoid division by zero if no overlap
#
#     # Apply mask to both images
#     image1_masked = image1.copy()
#     image2_masked = image2.copy()
#     image1_masked[~mask] = 0
#     image2_masked[~mask] = 0
#
#     # Compute SSIM only on masked region
#     ssim_index = ssim(
#         image1_masked, image2_masked,
#         data_range=255,  # assume 8-bit grayscale
#         win_size=11,
#         gradient=False,
#         full=False,
#         gaussian_weights=True
#     )
#     return ssim_index
#
#
# def calculate_mutual_information(image1, image2, visualization=False):
#     # Resize to match shape
#     if image1.shape != image2.shape:
#         image2 = resize(image2, image1.shape, preserve_range=True, anti_aliasing=True).astype(image1.dtype)
#
#     # Only consider valid (non-zero) pixels in both
#     mask = (image1 > 0) & (image2 > 0)
#
#     if np.sum(mask) == 0:
#         return 0
#
#     image1_masked = image1[mask]
#     image2_masked = image2[mask]
#
#     # Compute MI over masked pixels
#     hist_2d, _, _ = np.histogram2d(image1_masked.ravel(), image2_masked.ravel(), bins=20)
#     mi = mutual_info_score(None, None, contingency=hist_2d)
#     return mi



# Example usage

def calculate_mutual_information(image1, image2, visualization=False):
    hist_2d, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=20)
    mi = mutual_info_score(None, None, contingency=hist_2d)
    if visualization:
        # Plot the 2D histogram
        plt.figure(figsize=(8, 6))
        plt.imshow(hist_2d.T, origin='lower', aspect='auto',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        plt.colorbar(label='Counts')
        plt.xlabel('Image1 Pixel Intensities')
        plt.ylabel('Image2 Pixel Intensities')
        plt.title('2D Histogram of Joint Pixel Intensities')
        plt.show()

    return mi

def calculate_ssim(image1,image2, visualization=False):
    # Ensure the images are the same size
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Structural Similarity Index
    ssim_index, grad, s = ssim(image1, image2,gradient=True, full=True, multichannel=True)
    if visualization:
        plt.figure(figsize=(8, 6))
        plt.imshow(s, cmap='Purples')
        plt.colorbar(label='SSIM Value')
        plt.title('SSIM Map')
        plt.axis('off')  # Remove axis ticks for better visualization (optional)
        plt.show()

    return ssim_index



with_fiducial_path = '/media/huifang/data/fiducial/temp_result/cytassist/with_fiducial/'
without_fiducial_path = '/media/huifang/data/fiducial/temp_result/cytassist/without_fiducial/'

itk_image_with_marker="/media/huifang/data/fiducial/temp_result/vispro/registration/cytassist/with_marker/"
itk_image_marker_free="/media/huifang/data/fiducial/temp_result/vispro/registration/cytassist/marker_free/"


# Initialize accumulators
ssim_scores = {
    # 'Original (marker)': [],
    # 'Original (vispro)': [],
    'Registered ITK (marker)': [],
    'Registered ITK (vispro)': [],
    'Registered BJ (marker)': [],
    'Registered BJ (vispro)': []
}
mi_scores = {k: [] for k in ssim_scores}

for i in range(15):
    if i==2 or i==12:
        continue
    print(i)
    # Read all images
    fixed_original = cv2.imread(itk_image_with_marker+str(i)+"_fixed.png", cv2.IMREAD_GRAYSCALE)
    moving_original = cv2.imread(itk_image_with_marker + str(i) + "_moving.png", cv2.IMREAD_GRAYSCALE)
    itk_registered_original = cv2.imread(itk_image_with_marker + str(i) + "_registered.png", cv2.IMREAD_GRAYSCALE)
    bj_registered_original = cv2.imread(with_fiducial_path+str(i) + '/Registered Source Image.png', cv2.IMREAD_GRAYSCALE)

    fixed_vispro = cv2.imread(itk_image_marker_free + str(i) + "_fixed.png", cv2.IMREAD_GRAYSCALE)
    moving_vispro = cv2.imread(itk_image_marker_free + str(i) + "_moving.png", cv2.IMREAD_GRAYSCALE)
    itk_registered_vispro = cv2.imread(itk_image_marker_free + str(i) + "_registered.png", cv2.IMREAD_GRAYSCALE)
    bj_registered_vispro = cv2.imread(without_fiducial_path+str(i) + '/Registered Source Image.png', cv2.IMREAD_GRAYSCALE)


    # MI
    # mi_scores['Original (marker)'].append(calculate_mutual_information(fixed_original, moving_original))
    # mi_scores['Original (vispro)'].append(calculate_mutual_information(fixed_vispro, moving_vispro))
    mi_scores['Registered ITK (marker)'].append(calculate_mutual_information(fixed_original, itk_registered_original))
    mi_scores['Registered ITK (vispro)'].append(calculate_mutual_information(fixed_vispro, itk_registered_vispro))
    mi_scores['Registered BJ (marker)'].append(calculate_mutual_information(fixed_original, bj_registered_original))
    mi_scores['Registered BJ (vispro)'].append(calculate_mutual_information(fixed_vispro, bj_registered_vispro))

    # SSIM
    # ssim_scores['Original (marker)'].append(calculate_ssim(fixed_original, moving_original))
    # ssim_scores['Original (vispro)'].append(calculate_ssim(fixed_vispro, moving_vispro))
    ssim_scores['Registered ITK (marker)'].append(calculate_ssim(fixed_original, itk_registered_original))
    ssim_scores['Registered ITK (vispro)'].append(calculate_ssim(fixed_vispro, itk_registered_vispro))
    ssim_scores['Registered BJ (marker)'].append(calculate_ssim(fixed_original, bj_registered_original))
    ssim_scores['Registered BJ (vispro)'].append(calculate_ssim(fixed_vispro, bj_registered_vispro))
print(mi_scores['Registered BJ (marker)'])
print( mi_scores['Registered BJ (vispro)'])
test = input()
# Print averages
print("\n=== AVERAGED SSIM & MI RESULTS ===")
for method in ssim_scores.keys():
    avg_ssim = np.mean(ssim_scores[method])
    avg_mi = np.mean(mi_scores[method])
    print(f"{method:25s} - SSIM: {avg_ssim:.4f}, MI: {avg_mi:.4f}")

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte
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

def create_checkerboard(image1, image2, num_squares=80):
    checkerboard = np.zeros_like(image1)
    h, w = image1.shape[:2]
    s_h, s_w = h // num_squares, w // num_squares

    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2:
                checkerboard[i*s_h:(i+1)*s_h, j*s_w:(j+1)*s_w] = image1[i*s_h:(i+1)*s_h, j*s_w:(j+1)*s_w]
            else:
                checkerboard[i*s_h:(i+1)*s_h, j*s_w:(j+1)*s_w] = image2[i*s_h:(i+1)*s_h, j*s_w:(j+1)*s_w]
    return checkerboard

def compute_residual(image1, image2):
    residual = cv2.absdiff(image1, image2)
    return np.mean(residual), np.std(residual)

def show_image(title, image1,image2):
    f,a = plt.subplots(1,2,figsize=(20,10))
    a[0].imshow(image1, cmap='gray')
    a[1].imshow(image2,cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

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


def calculate_correlation_coefficient(image1, image2):
    # Ensure the images are the same size
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Correlation Coefficient
    correlation_coefficient = np.corrcoef(image1.flatten(), image2.flatten())[0, 1]

    return correlation_coefficient

def read_images(path,i,id1,id2,):
    img_path1 = path + str(i) + '/' + str(id1) + '.png'
    img_path2 = path + str(i)+'/'+str(id2)+'.png'
    registered_img_path1 = path + str(i) + '/Registered Source Image.png'
    registered_img_path2 = path + str(i) + '/Registered Target Image.png'


    image1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    registered_image1 = cv2.imread(registered_img_path1, cv2.IMREAD_GRAYSCALE)
    registered_image2 = cv2.imread(registered_img_path2, cv2.IMREAD_GRAYSCALE)
    return image1,image2,registered_image1,registered_image2

def read_images2(path,i):
    img_path1 = path + str(i) + '/' + str(i) + '_without_fiducial.png'
    img_path2 = path + str(i)+'/'+str(i)+'_tissue.png'
    registered_img_path1 = path + str(i) + '/Registered Source Image.png'
    registered_img_path2 = path + str(i) + '/Registered Target Image.png'


    image1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    registered_image1 = cv2.imread(registered_img_path1, cv2.IMREAD_GRAYSCALE)
    registered_image2 = cv2.imread(registered_img_path2, cv2.IMREAD_GRAYSCALE)
    return image1,image2,registered_image1,registered_image2

def read_images_with_fiducial(path,i):

    img_path1 = path + str(i) + '/' + str(i) + '_with_fiducial.png'
    img_path2 = path + str(i)+'/'+str(i)+'_tissue.png'
    registered_img_path1 = path + str(i) + '/Registered Source Image.png'
    registered_img_path2 = path + str(i) + '/Registered Target Image.png'

    image1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    registered_image1 = cv2.imread(registered_img_path1, cv2.IMREAD_GRAYSCALE)
    registered_image2 = cv2.imread(registered_img_path2, cv2.IMREAD_GRAYSCALE)
    return image1,image2,registered_image1,registered_image2

def get_separate_images(path,i):
    image1 = cv2.imread(path + str(i)+'_m_w_f.png', cv2.IMREAD_GRAYSCALE)
    # m = image1[:, :1024]
    # w = image1[:, 1024:2048]
    # f = image1[:, 2048:]
    m = image1[:, :1408]
    w = image1[:, 1408:2816]
    f = image1[:, 2816:]
    # plt.imshow(f)
    # plt.show()


    return m,w,f

# def get_blended_image(image1,image2):
#     alpha=0.5
#         # Create a new array with the maximum shape
#     max_shape = (max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1]))
#     blended_image = np.zeros(max_shape, dtype=np.float64)
#
#     # Place img1 and img2 into overlay_img
#     blended_image[:image1.shape[0], :image1.shape[1]] += alpha * image1
#     blended_image[:image2.shape[0], :image2.shape[1]] += (1 - alpha) * (1-image2)
#     return blended_image
def get_blended_image(image1, image2):
    alpha = 0.5

    # Calculate the minimum shape
    min_shape = (min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1]))

    # Resize images to the minimum shape
    resized_image1 = resize(image1, min_shape, anti_aliasing=True, preserve_range=True)
    resized_image2 = resize(image2, min_shape, anti_aliasing=True, preserve_range=True)

    # Blend the resized images
    blended_image = alpha * (resized_image1) + (1 - alpha) * ( resized_image2)
    return blended_image






# Example usage
with_fiducial_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/cytassist/with_fiducial/'
without_fiducial_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/cytassist/without_fiducial/'

# with_fiducial_path='/home/huifang/workspace/code/fiducial_remover/temp_result/application/registration/with_fiducial/'
# without_fiducial_path ='/home/huifang/workspace/code/fiducial_remover/temp_result/application/registration/without_fiducial/'

vxm_path = '/home/huifang/workspace/code/voxelmorph/results/without_fiducial/all/'

metric_with_fiducial=0
metric_without_fiducial=0
for i in range(0,15):
    print(i)
    if i ==2:
        continue
    image_id = registration_pair[i]
    id1 = image_id[0]
    id2 = image_id[1]

    # m,w,f = get_separate_images(vxm_path,i)
    # metric_without_fiducial += calculate_ssim(w,f, visualization=True)
    # continue


    image1_nf, image2_nf, registered_image1_nf, registered_image2_nf = read_images2(without_fiducial_path, i)

    # metric_with_fiducial +=calculate_ssim(image2, registered_image2)

    # metric_without_fiducial += calculate_ssim(image1_nf, registered_image1_nf, visualization=False)


    image1,image2,registered_image1,registered_image2 = read_images_with_fiducial(with_fiducial_path,i)


    # metric_with_fiducial += calculate_ssim(image1, registered_image1, visualization=False)

    blended_image1 = get_blended_image(image1_nf,registered_image1_nf)
    blended_image2 = get_blended_image(image1,registered_image1)
    # outpath = '/home/huifang/workspace/code/fiducial_remover/temp_result/application/registration/figures/'
    # plt.imsave(outpath +'2cytassist_'+ str(i) + '_blended_image1.png', blended_image1, cmap='gray')
    # plt.imsave(outpath +'2cytassist_'+ str(i) + '_blended_image2.png', blended_image2, cmap='gray')
    f,a =plt.subplots(1,2,figsize=(20,10))
    a[0].imshow(blended_image1, cmap='gray')  # Use a grayscale colormap for display
    a[1].imshow(blended_image2, cmap='gray')  # Use a grayscale colormap for display
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

    print('saved')
    # test = input()




    # metric_without_fiducial += calculate_ssim(image2_nf, registered_image2_nf)
    # m1, w1, f1 = get_separate_images(with_fiducial_path, i)
    # m2, w2, f2 = get_separate_images(without_fiducial_path, i)
    # metric_with_fiducial += calculate_ssim(w1,f1)
    # metric_without_fiducial += calculate_ssim(w2, f2)



print(metric_with_fiducial/14)
print(metric_without_fiducial/14)
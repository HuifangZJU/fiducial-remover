import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from matplotlib import pyplot as plt
from skimage.transform import resize
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


# Example usage
with_fiducial_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/cytassist/with_fiducial/'
without_fiducial_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/cytassist/without_fiducial/'

# with_fiducial_path='/home/huifang/workspace/code/fiducial_remover/temp_result/application/registration/with_fiducial/'
# without_fiducial_path ='/home/huifang/workspace/code/fiducial_remover/temp_result/application/registration/without_fiducial/'

vxm_path = '/home/huifang/workspace/code/voxelmorph/results/without_fiducial/all/'

itk_image_with_marker="/media/huifang/data/fiducial/temp_result/vispro/registration/with_marker/"
itk_image_marker_free="/media/huifang/data/fiducial/temp_result/vispro/registration/marker_free/"

metric_with_fiducial=0
metric_without_fiducial=0
for i in range(0,15):
    print(i)
    if i ==2:
        continue
    image_id = registration_pair[i]
    id1 = image_id[0]
    id2 = image_id[1]


    fixed_image_with_marker = itk_image_with_marker+str(i)+"_fixed.png"
    moving_image_with_marker = itk_image_with_marker + str(i) + "_moving.png"
    itk_registered_image_with_marker = itk_image_with_marker + str(i) + "_registered.png"
    bj_registered_image_with_marker = with_fiducial_path+str(i) + '/Registered Source Image.png'

    fixed_image_marker_free = itk_image_marker_free + str(i) + "_fixed.png"
    moving_image_marker_free = itk_image_marker_free + str(i) + "_moving.png"
    itk_registered_image_marker_free = itk_image_marker_free + str(i) + "_registered.png"
    bj_registered_image_marker_free = without_fiducial_path+str(i) + '/Registered Source Image.png'





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
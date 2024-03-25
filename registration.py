from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np
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

def save_rgb_image(array, filename):
    # The input is expected to be in the range [0, 1], so we scale it to [0, 255]
    if np.max(array)<1.1:
        array = 255*array
    array = array.astype('uint8')
    cv2.imwrite(filename, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR format

def get_gray_image(path):
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
def get_blended_image(img_path1,img_path2):
    # registered_image1 = cv2.imread(img_path1)
    # registered_image2 = cv2.imread(img_path2)

    # # Blend the two images (adjust the alpha value for the desired transparency effect)
    # alpha = 0.5  # Adjust this value as needed
    # blended_image = cv2.addWeighted(registered_image1, alpha, registered_image2, 1 - alpha, 0)

    # visualization
    image1 = get_gray_image(img_path1)
    image2 = get_gray_image(img_path2)

    # image1 = image1 / 255
    # image2 = image2 / 255
    # plt.imshow(image1,cmap='gray')
    # plt.imshow(image2,cmap='gray',alpha=0.5)
    # plt.show()

    # # Blend the two adjusted images to overlay them
    alpha = 0.5  # Adjust this value to control the blending transparency
    blended_image = alpha* image1 + (1 - alpha) * (1-image2)
    return blended_image

test_image_path = '/home/huifang/workspace/data/imagelists/fiducial_previous/st_image_trainable_fiducial.txt'
f = open(test_image_path, 'r')
files = f.readlines()
# f.close()
# num_files = len(files)

# registration_image_list =  '/home/huifang/workspace/data/imagelists/UnwarpJ_cytassist_with_fiducial_registration_list.txt'
# image_path = '/home/huifang/workspace/code/backgroundremover/bgrm_out/'
# image_path = '/home/huifang/workspace/code/fiducial_remover/temp_result/registration/with_fiducial/'
image_path1 = '/home/huifang/workspace/code/fiducial_remover/temp_result/cytassist/with_fiducial/'
image_path2 = '/home/huifang/workspace/code/fiducial_remover/temp_result/cytassist/without_fiducial/'
# f_list = open(registration_image_list,'w')
for i in range(0,15):
    if i==2:
        continue
    # image_id = registration_pair[i]
    # id1 = image_id[0]
    # id2 = image_id[1]
    # # img_path1 = files[id1].split(' ')[0]
    # # img_path2 = files[id2].split(' ')[0]
    print(i)
    img_path1 = image_path1 + str(i) + '/' + str(i) + '_with_fiducial_resized.png'
    # img_path2 = image_path1+str(i)+'/'+str(i)+'_tissue.png'
    img_path2 = image_path2 + str(i) + '/' + str(i) + '_without_fiducial_resized.png'

    registered_img_path1 = image_path1+str(i)+'/Registered Source Image.png'
    # registered_img_path2 = image_path1 + str(i) + '/Registered Target Image.png'
    registered_img_path2 = image_path2 + str(i) + '/Registered Source Image.png'
    # f_list.write(registered_img_path1 + ',' + img_path1 + '\n')
    # f_list.write(registered_img_path2 + ',' + img_path2 + '\n')
    # continue
    # image1 = plt.imread(img_path1)
    # image2 = plt.imread(registered_img_path1)
    # #
    # plt.imshow(image1)
    # plt.imshow(image2,alpha=0.5)
    # plt.show()


    blended_image1 = get_blended_image(img_path1, registered_img_path1)
    blended_image2 = get_blended_image(img_path2, registered_img_path2)
    f,a =plt.subplots(1,2,figsize=(20,10))
    a[0].imshow(blended_image1, cmap='gray')  # Use a grayscale colormap for display
    a[1].imshow(blended_image2, cmap='gray')  # Use a grayscale colormap for display
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

# f_list.close()

# registration_image_list = '/home/huifang/workspace/data/imagelists/st_cytassist.txt'
# registration_pairs = open(registration_image_list,'r')
#
# for line in registration_pairs:
#     img_path1 = line.split(' ')[0]
#     img_path2 = line.split(' ')[1].rstrip('\n')
#     img1 = plt.imread(img_path1)
#     img2 = plt.imread(img_path2)
#     f,a = plt.subplots(1,2)
#     a[0].imshow(img1)
#     a[1].imshow(img2)
#     plt.show()


from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import os
import json
def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr



def plot_detail_polygon(json_path,image):
    with open(json_path, 'r') as file:
        data = json.load(file)
    for j, shape in enumerate(data['shapes']):
        if shape['label'] == 'detail' and shape['shape_type'] == 'polygon':
            points = shape['points']
            points_np = np.array(points, np.int32)
            # Reshape the points array to the shape (number_of_vertices, 1, 2)
            points_np = points_np.reshape((-1, 1, 2))
            # Draw the polygon on the image
            cv2.polylines(image, [points_np], isClosed=True, color=(0, 255, 0), thickness=2)
    return image









def get_separate_images(path,i):
    image1 = mpimg.imread(path + str(i)+'_m_w_f.png')
    if os.path.exists(path + str(i)+'_m_w_f.json'):
        image1 = plot_detail_polygon(path + str(i)+'_m_w_f.json',image1)
    # m = image1[:, :1024, :]
    # w = image1[:, 1024:2048, :]
    # f = image1[:, 2048:, :]
    m = image1[:, :1408, :]
    w = image1[:, 1408:2816, :]
    f = image1[:, 2816:, :]
    return m,w,f
def get_blended_image(image1,image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image1 = image1/255
    image2 = image2/255
    alpha = 0.5  # Adjust this value to control the blending transparency
    blended_image = alpha * image1 + (1 - alpha) * (1-image2)
    blended_image = normalize_array(blended_image)
    return blended_image


# image_path1 = '/home/huifang/workspace/code/fiducial_remover/temp_result/registration/bUnwarpJ/with_fiducial/'
# image_path2 = '/home/huifang/workspace/code/fiducial_remover/temp_result/registration/bUnwarpJ/without_fiducial/'
image_path1='/home/huifang/workspace/code/voxelmorph/results/with_fiducial/all/'
image_path2='/home/huifang/workspace/code/voxelmorph/results/without_fiducial/all/'



for i in range(25,40):
    print(i)
    #
    # img1 = plt.imread(image_path1+str(i)+'.png')
    # img2 = plt.imread(image_path2+str(i)+'-good.png')
    # fig,a = plt.subplots(1,2,figsize=(20, 10))
    # a[0].imshow(img1)
    # a[1].imshow(img2)



    #
    m1,w1,f1 = get_separate_images(image_path1, i)
    m2,w2,f2 = get_separate_images(image_path2, i)

    if i%2 == 0:
        _, _, mm1 = get_separate_images(image_path1, i+1)
        _, _, mm2 = get_separate_images(image_path2, i+1)
    else:
        _, _, mm1 = get_separate_images(image_path1, i-1)
        _, _, mm2 = get_separate_images(image_path2, i-1)





    fig1,a1 = plt.subplots(2,4,figsize=(40, 20))
    a1[0, 0].imshow(mm1)
    a1[0, 1].imshow(m1)
    a1[0, 2].imshow(w1)
    a1[0, 3].imshow(f1)
    # blended1 = get_blended_image(f1,w1)
    # blended2 = get_blended_image(f2, w2)
    # a1[1, 1].imshow(blended1, cmap='gray')
    # a1[1, 2].imshow(blended2, cmap='gray')

    # fig2, a2 = plt.subplots(2, 3)
    a1[1, 0].imshow(mm2)
    a1[1, 1].imshow(m2)
    a1[1, 2].imshow(w2)
    a1[1, 3].imshow(f2)
    # blended2 = get_blended_image(f2, w2)
    # a2[1, 1].imshow(blended2, cmap='gray')
    plt.show()
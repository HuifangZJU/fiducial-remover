from __future__ import division
import os.path
import random
import matplotlib.pyplot as plt

from hough_utils import *
from fiducial_utils import read_tissue_image as read_image

crop_size = 16
augsize =6
circle_width = 2

def write_image_in_dir_to_file(root_dir,f,ds_rate,save_in_tissue=True,save_out_tissue=True):
    image_names = os.listdir(root_dir)
    for image_name in image_names:
        dir = root_dir + image_name+'/'
        if save_in_tissue:
            in_tissue_images = os.listdir(dir+'in_tissue')
            for crop_image_name in in_tissue_images:
                if crop_image_name.endswith('image.png'):
                    crop_image = dir+'in_tissue/'+crop_image_name
                    crop_image_mask = crop_image[:-9]+'mask'+crop_image[-4:]
                    f.write(crop_image+' '+crop_image_mask + '\n')
        if save_out_tissue:
            out_tissue_images = os.listdir(dir + 'out_tissue')
            i = 0
            for crop_image_name in out_tissue_images:
                if crop_image_name.endswith('image.png'):
                    i += 1
                    if np.mod(i,ds_rate)!=0:
                        continue
                    crop_image = dir + 'out_tissue/' + crop_image_name
                    crop_image_mask = crop_image[:-9] + 'mask' + crop_image[-4:]
                    f.write(crop_image + ' ' + crop_image_mask + '\n')
    # print(dir +' done')


def write_to_file(image_root_path,save_file,downsample_rate,save_in_tissue=True,save_out_tissue=True):

    f = open(save_file,'w')
    image_second_paths = os.listdir(image_root_path)
    for image_path in image_second_paths:
        temp_path = image_root_path + image_path + '/'
        write_image_in_dir_to_file(temp_path,f,downsample_rate,save_in_tissue,save_out_tissue)






def save_local_crop(image,mask,circle,path,iter_num=1,augmentation=True):

    for i in range(iter_num):
        xc = circle[0]
        yc = circle[1]
        if augmentation:
            x_rand = random.randint(-augsize, augsize)
            y_rand = random.randint(-augsize, augsize)
            xc = xc+x_rand
            yc = yc+y_rand

        crop_image = image[yc - crop_size:yc + crop_size, xc - crop_size:xc + crop_size, :]
        crop_mask = mask[yc - crop_size:yc + crop_size, xc - crop_size:xc + crop_size]
        # f,a = plt.subplots(1,2)
        # a[0].imshow(crop_image)
        # a[1].imshow(crop_mask)
        # plt.show()
        save_image(crop_image, path + str(yc) + '_' + str(xc) + '_'+ str(i)+'_image.png',format="RGB")
        save_image(crop_mask, path + str(yc) + '_' + str(xc) + '_'+ str(i)+'_mask.png', format="L")


# ------------------------------------------
#                data loading
# ------------------------------------------

root_path = '/home/huifang/workspace/data/fiducial_train/'
save_root = '/home/huifang/workspace/data/fiducial_crop_w2_aug'+str(augsize)+'_iter4/'
dataset_names = os.listdir(root_path)
in_tissue_all=0
out_tissue_all=0
for dataset_name in dataset_names:
    print('-------------'+dataset_name+'-------------')
    image_names= os.listdir(root_path+dataset_name)
    for image_name in image_names:
        print(image_name+'...')
        image = read_image(root_path+dataset_name+'/' + image_name)
        circles = np.load(root_path+dataset_name+'/' + image_name+'/masks/human_in_loop_mask.npy')
        mask = generate_mask(image.shape[:2],circles,circle_width)
        in_tissue = np.where(circles[:,-1]==1)
        out_tissue = np.where(circles[:,-1]==0)

        in_tissue_save_path = save_root+dataset_name+'/' + image_name +'/in_tissue/'
        out_tissue_save_path = save_root + dataset_name + '/' + image_name + '/out_tissue/'
        os.makedirs(in_tissue_save_path,exist_ok=True)
        os.makedirs(out_tissue_save_path,exist_ok=True)

        for id in in_tissue[0]:
            save_local_crop(image, mask, circles[id,:],in_tissue_save_path,iter_num=4)
        for id in out_tissue[0]:
            save_local_crop(image, mask, circles[id, :], out_tissue_save_path,iter_num=4)
        print('done')
        in_tissue_all += len(in_tissue[0])
        out_tissue_all += len(out_tissue[0])
print(in_tissue_all)
print(out_tissue_all)

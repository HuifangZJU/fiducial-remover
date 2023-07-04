import sys

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import matplotlib.pyplot as plt

SAVE_ROOT = '/media/huifang/data/fiducial/annotation/'

def get_annotated_circles(annotation_path):
    in_tissue_path = os.path.join(annotation_path, 'in_tissue')
    in_tissue_circle = [circle for circle in os.listdir(in_tissue_path) if circle.endswith('image.png')]
    in_circle_meta = [[int(u), int(v), int(r), 1] for circle in in_tissue_circle for v, u, r, _ in [circle.split('_')]]

    out_tissue_path = os.path.join(annotation_path, 'auto')
    out_tissue_circle = [circle for circle in os.listdir(out_tissue_path) if circle.endswith('image.png')]
    out_circle_meta = [[int(u), int(v), int(r), 0] for circle in out_tissue_circle for v, u, r, _ in [circle.split('_')]]

    return in_circle_meta, out_circle_meta

def annotate_patches(image_size, step, circles):
    w,h = image_size
    num_patches_w = w // step
    num_patches_h = h // step

    annotation = np.zeros((num_patches_w, num_patches_h), dtype=float)
    image_mask = np.zeros(image_size)

    for i in range(num_patches_w):
        for j in range(num_patches_h):
            patch_x = i * step
            patch_y = j * step
            patch_rect = (patch_x, patch_y, step, step)

            for circle in circles:
                circle_x, circle_y, circle_radius = circle[:3]
                circle_radius = circle_radius+2
                circle_rect = (circle_y - circle_radius,circle_x - circle_radius,  2 * circle_radius, 2 * circle_radius)


                if rectangles_intersect(patch_rect, circle_rect):
                    annotation[i, j] = 1.0
                    image_mask[patch_x:patch_x+step,patch_y:patch_y+step]=1
                    # plt.imshow(image[patch_x:patch_x + step, patch_y:patch_y + step])
                    # print(annotation[i, j])
                    # plt.show()
                    break


    return annotation, image_mask

def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)



def get_annotation_path(imagepath):
    dataset = imagepath.split('/')[6]
    index = imagepath.split('/')[7]
    index = index.split('.')
    index = index[0]
    data_path = SAVE_ROOT + dataset + '_' + index
    return data_path


class ImageDataset(Dataset):
    def __init__(self, path, transforms_a=None,transforms_b=None):
        self.transform_a = transforms.Compose(transforms_a)
        self.transform_b = transforms.Compose(transforms_b)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_a = Image.open(img_path[0])
        img_b = Image.open(img_path[1])


        if np.random.random() < 0.7:
            if np.random.random()<0.5:
                img_a = Image.fromarray(np.fliplr(np.array(img_a)),'RGB')
                img_b = Image.fromarray(np.fliplr(np.array(img_b)), 'L')
            else:
                img_a = Image.fromarray(np.flipud(np.array(img_a)),'RGB')
                img_b = Image.fromarray(np.flipud(np.array(img_b)), 'L')

        img_a = self.transform_a(img_a)
        img_b = self.transform_b(img_b)
        return {'A': img_a, 'B':img_b}

    def __len__(self):
        return len(self.files)

class BinaryDataset(Dataset):
    def __init__(self, transforms_=None,patch_size=32):
        self.transform = transforms.Compose(transforms_)
        self.patch_size = patch_size
        path ='/home/huifang/workspace/data/imagelists/st_image_trainable_temp_fiducial.txt'
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):

        image_name = self.files[index % len(self.files)]
        image_name = image_name[:-1]
        img_a = Image.open(image_name)
        # show_grids(image,64)
        annotation_path = get_annotation_path(image_name)
        in_tissue_circles, out_tissue_circles = get_annotated_circles(annotation_path)
        circles = in_tissue_circles + out_tissue_circles
        h,w = img_a.size
        patches, annotation_image = annotate_patches([w,h], self.patch_size, circles)
        # if np.random.random() < 0.7:
        #     if np.random.random()<0.5:
        #         img_a = Image.fromarray(np.fliplr(np.array(img_a)),'RGB')
        #         patches = Image.fromarray(np.fliplr(np.array(patches)), 'L')
        #     else:
        #         img_a = Image.fromarray(np.flipud(np.array(img_a)),'RGB')
        #         patches = Image.fromarray(np.flipud(np.array(patches)), 'L')
        img_a = self.transform(img_a)
        patches = np.array(patches,dtype=np.float32)
        return {'A': img_a, 'B':patches}

    def __len__(self):
        return len(self.files)

class ImageRandomCropDataset(Dataset):
    def __init__(self, path, crop_size,transforms_a=None,transforms_b=None):
        self.transform_a = transforms.Compose(transforms_a)
        self.transform_b = transforms.Compose(transforms_b)
        self.crop_size = crop_size
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_a = Image.open(img_path[0])
        img_b = Image.open(img_path[1])
        w,h = img_a.size
        crop_w_start = np.random.randint(0,w-self.crop_size)
        crop_h_start = np.random.randint(0,h-self.crop_size)

        img_a = img_a.crop((crop_w_start, crop_h_start, self.crop_size, self.crop_size))
        img_b = img_b.crop((crop_w_start, crop_h_start, self.crop_size, self.crop_size))

        print(img_a.size)
        print(img_b.size)
        test = input()


        if np.random.random() < 0.7:
            if np.random.random()<0.5:
                img_a = Image.fromarray(np.fliplr(np.array(img_a)),'RGB')
                img_b = Image.fromarray(np.fliplr(np.array(img_b)), 'L')
            else:
                img_a = Image.fromarray(np.flipud(np.array(img_a)),'RGB')
                img_b = Image.fromarray(np.flipud(np.array(img_b)), 'L')

        img_a = self.transform_a(img_a)
        img_b = self.transform_b(img_b)
        return {'A': img_a, 'B':img_b}

    def __len__(self):
        return len(self.files)

class ImageTestDataset(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_a = Image.open(img_path[0])
        if np.random.random() < 0.7:
            if np.random.random()<0.5:
                img_a = Image.fromarray(np.fliplr(np.array(img_a)),'RGB')
            else:
                img_a = Image.fromarray(np.flipud(np.array(img_a)),'RGB')

        img_a = self.transform(img_a)
        return {'A': img_a}

    def __len__(self):
        return len(self.files)
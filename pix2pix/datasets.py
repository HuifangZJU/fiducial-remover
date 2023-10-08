import sys


from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import cv2
from matplotlib import pyplot as plt

SAVE_ROOT = '/media/huifang/data/fiducial/annotation/'

def get_augmentation_parameters():
    corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    preserved_corner = np.random.choice(corners)

    # Define cropping ranges based on the preserved corner
    crop_range = 0.2
    if preserved_corner == 'top-left':
        crop_values = ((0, 0), (0, 0), (0, crop_range), (0, crop_range))
    elif preserved_corner == 'top-right':
        crop_values = ((0, 0), (0, crop_range), (0, crop_range), (0, 0))
    elif preserved_corner == 'bottom-left':
        crop_values = ((0, crop_range), (0, 0), (0, 0), (0, crop_range))
    else:  # 'bottom-right'
        crop_values = ((0, crop_range), (0, crop_range), (0, 0), (0, 0))

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),
        # iaa.Crop(percent=crop_values),
        iaa.Affine(
            scale=(0.8, 1.0),  # random scale between 80% and 100%
            rotate=(-15, 15)  # random rotation between -25 to 25 degrees
        ),
        iaa.Multiply((0.7, 1.3)),  # change brightness
        iaa.GaussianBlur(sigma=(0, 2.0))  # apply Gaussian blur
    ], random_order=False)  # apply the augmentations in random order
    return seq



def augment_image_and_keypoints(image, keypoints):
    image = image.astype(np.uint8)
    seq = get_augmentation_parameters()
    # Convert your [x, y, r] format to Keypoint format for imgaug
    kps = [ia.Keypoint(x=p[0], y=p[1]) for p in keypoints]
    kps_obj = ia.KeypointsOnImage(kps, shape=image.shape)

    # Augment image and keypoints
    image_aug, kps_aug = seq(image=image, keypoints=kps_obj)
    scale_factor = image_aug.shape[1] / image.shape[1]  # based on width
    keypoints_aug = [(kp.x, kp.y, p[2] * scale_factor) for kp, p in zip(kps_aug.keypoints, keypoints)]
    keypoints_aug = [(int(x), int(y),int(z)) for x, y, z in keypoints_aug]

    return image_aug, keypoints_aug

def augment_image_and_mask(image, mask):
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)
    seq = get_augmentation_parameters()
    image_aug, mask_aug = seq(images=[image],segmentation_maps=[mask])
    return image_aug[0],mask_aug[0]


def convert_cv2_to_pil(cv2_img):
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image and return
    pil_image = Image.fromarray(rgb_image)
    return pil_image
def convert_pil_to_cv2(pil_img):
    np_image = np.array(pil_img)

    # Convert the NumPy array to a cv2 image
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return cv2_image

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
                    break
    return annotation

def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def annotate_dots(image_size, circles):
    w,h = image_size
    annotation = np.zeros((w, h), dtype=float)


    for circle in circles:
        circle_x, circle_y, circle_radius = circle[:3]
        cv2.circle(annotation,(circle_x,circle_y),circle_radius,1,thickness=-1)
    return annotation

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
    def __init__(self, transforms_=None,patch_size=32,mode='test',test_group=1,aug=True):
        self.transform = transforms.Compose(transforms_)
        self.patch_size = patch_size
        self.mode = mode
        self.aug = aug
        path ='/home/huifang/workspace/data/imagelists/st_image_trainable_temp_fiducial.txt'
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        self.files=[]
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            if self.mode == 'train' and int(group) != test_group:
                self.files.append(line)
            if self.mode == 'test' and int(group) == test_group:
                self.files.append(line)

    def __getitem__(self, index):

        image_name = self.files[index % len(self.files)].split(' ')[0]
        image_name = image_name.rstrip('\n')
        img_a = cv2.imread(image_name)
        # show_grids(image,64)
        annotation_path = get_annotation_path(image_name)
        circles = np.load(annotation_path + '/circles.npy')
        if self.mode == 'train':
            img_a, circles = augment_image_and_keypoints(img_a, circles)
        img_a = convert_cv2_to_pil(img_a)
        h,w = img_a.size
        patches= annotate_patches([w,h], self.patch_size, circles)
        img_a = self.transform(img_a)
        patches = np.array(patches,dtype=np.float32)
        return {'A': img_a, 'B':patches}

    def __len__(self):
        return len(self.files)

def find_nearest_multiple_of_32(x):
    base = 32
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)
class DotDataset(Dataset):
    def __init__(self, transforms_=None,mode='test',test_group=1,aug=True):
        self.transform = transforms.Compose(transforms_)
        self.mode =mode
        self.aug = aug
        path ='/home/huifang/workspace/data/imagelists/st_image_trainable_temp_fiducial.txt'
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        self.files=[]
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            if self.mode == 'train' and int(group) != test_group:
                self.files.append(line)
            if self.mode == 'test' and int(group) == test_group:
                self.files.append(line)

    def __getitem__(self, index):

        image_name = self.files[index % len(self.files)].split(' ')[0]
        image_name = image_name.rstrip('\n')
        img_a = Image.open(image_name)
        # img_a = cv2.imread(image_name)
        # show_grids(image,64)
        annotation_path = get_annotation_path(image_name)
        circles = np.load(annotation_path+'/circles.npy')
        h,w = img_a.size
        h_new = find_nearest_multiple_of_32(h)
        w_new = find_nearest_multiple_of_32(w)
        circles = np.array(circles)
        circles[:,0] = circles[:,0]*h_new/h
        circles[:, 1] = circles[:, 1] * w_new / w
        annotation = annotate_dots([w_new,h_new],circles)
        annotation = annotation.reshape(annotation.shape[0],annotation.shape[1],1)
        img_a = img_a.resize((h_new, w_new), Image.ANTIALIAS)
        if self.mode == 'train':
            img_a = convert_pil_to_cv2(img_a)
            img_a, annotation = augment_image_and_mask(img_a,annotation)
            img_a = convert_cv2_to_pil(img_a)
        f,a = plt.subplots(1,2)
        a[0].imshow(img_a)
        a[1].imshow(annotation)
        plt.show()
        annotation = np.squeeze(annotation)
        img_a = self.transform(img_a)
        annotation = np.array(annotation,dtype=np.float32)
        return {'A': img_a, 'B':annotation}

    def __len__(self):
        return len(self.files)



class PointsDataset(Dataset):
    def __init__(self, transforms_=None,mode='test',test_group=1,aug=True):
        self.transform = transforms.Compose(transforms_)
        self.aug = aug
        path ='/home/huifang/workspace/data/imagelists/st_image_trainable_temp_fiducial.txt'
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        self.files=[]
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            if mode == 'train' and int(group) != test_group:
                self.files.append(line)
            if mode == 'test' and int(group) == test_group:
                self.files.append(line)

    def __getitem__(self, index):

        image_name = self.files[index % len(self.files)].split(' ')[0]
        image_name = image_name.rstrip('\n')
        # img_a = Image.open(image_name)
        img_a = cv2.imread(image_name)
        # show_grids(image,64)
        annotation_path = get_annotation_path(image_name)
        circles = np.load(annotation_path+'/circles.npy')

        img_a,circles = augment_image_and_keypoints(img_a,circles)
        plt.imshow(img_a)
        plt.show()
        h,w = img_a.shape[:2]
        h_new = find_nearest_multiple_of_32(h)
        w_new = find_nearest_multiple_of_32(w)
        circles = np.array(circles)
        circles[:, 0] = circles[:, 0]*h_new/h
        circles[:, 1] = circles[:, 1] * w_new /w
        img_a = Image.fromarray(img_a)
        img_a = img_a.resize((h_new,w_new), Image.ANTIALIAS)
        img_a = self.transform(img_a)
        return {'A': img_a, 'B':circles}

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
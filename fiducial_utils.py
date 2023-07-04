from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
import seaborn as sns
from hough_utils import *
F_RADIUS = 15

def get_fiducial_template():
    path = '/home/huifang/workspace/data/mouse/posterior_v1/spatial/aligned_fiducials.jpg'
    img = plt.imread(path)

    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    img_g = np.where(img_g < 10, img_g, 255 * np.ones(img_g.shape))
    img_b = np.where(img_b < 10, img_b, 255 * np.ones(img_b.shape))
    plt.imshow(img_b, cmap='gray', vmin=0, vmax=255)
    plt.show()

    fiducials = Image.fromarray(img_g)
    fiducials = fiducials.convert('RGB')
    fiducials.save('fiducials.jpeg')


def read_tissue_image(imgpath):
    try:
        image = plt.imread(imgpath + '/tissue_hires_image.png')
    except:
        try:
            image = plt.imread(imgpath + '/spatial/tissue_hires_image.png')
        except:
            print("Cannot find tissue_hires_image in path " + imgpath + " !")
            return
    return image

def save_image(array,filename,format="RGB"):
    if array.max()<1.1:
        array = 255 * array
    array = array.astype(np.uint8)
    array = Image.fromarray(array)
    if format == "RGB":
        array = array.convert('RGB')
    else:
        assert(format == "L")
        array = array.convert('L')
    array.save(filename)


def get_aligned_fiducial(path):
    img = plt.imread(path)
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]
    img_r = np.where(img_r > 180, np.ones(img_g.shape), 0)
    img_g = np.where(img_g < 30, np.ones(img_g.shape), 0)
    img_b = np.where(img_b < 30, np.ones(img_g.shape), 0)
    img_f = img_r*img_g*img_b
    scale = np.where(img_f==1)
    scale_u = max(scale[0]) - min(scale[0])
    scale_v = max(scale[1]) - min(scale[1])
    scale = min(scale_v/img.shape[0],scale_u/img.shape[1])
    # scale = 0.5*(scale_v/img.shape[0]+scale_u/img.shape[1])
    img_f = Image.fromarray(255-255*img_f)
    img_f = img_f.convert('RGB')
    img_f = np.asarray(img_f)
    # f,a = plt.subplots(1,2,figsize=(50, 50))
    # a[0].imshow(img)
    # a[1].imshow(img_f)
    # plt.show()
    return img_f, scale

def mouse_para():
    circle_path = './template/mouse_circles_f.txt'
    square_path = './template/mouse_square_f.txt'

    if not os.path.exists(circle_path):
        circles_f, framecenter_x, framecenter_y, square_scale = runSquare(
            '/home/huifang/workspace/data/mouse/posterior_v1/spatial/aligned_fiducials.jpg')
        np.savetxt(circle_path, circles_f)
        np.savetxt(square_path, np.array([framecenter_x, framecenter_y, square_scale]))
    else:
        circles_f = np.loadtxt(circle_path)
        [framecenter_x, framecenter_y, square_scale] = np.loadtxt(square_path)
    return circles_f, framecenter_x, framecenter_y, square_scale

def runCircle(fiducial_path):
    aligned_fiducials, scale = get_aligned_fiducial(fiducial_path)
    radius = round(scale*16)
    threshold = round(scale*40)
    circles_f = run_circle_threhold(aligned_fiducials, radius, circle_threshold=threshold, step=2)
    # for i in range(circles_f.shape[0]):
    #     cv2.circle(aligned_fiducials, (circles_f[i, 0], circles_f[i, 1]), circles_f[i, 2], (0, 255, 0), 1)
    # plt.figure(figsize=(15,15))
    # plt.imshow(aligned_fiducials)
    # plt.show()
    return circles_f,scale

def runSquare(fiducial_path):
    fiducials = get_aligned_fiducial(fiducial_path)
    circles_f = run_circle_threhold(fiducials,F_RADIUS,circle_threshold=50,step=2)
    ver_lines_upper,ver_lines_lower,hor_lines_upper,hor_lines_lower = get_square_lines(fiducials.shape[0],fiducials.shape[1],circles_f)
    ver_min = np.min(ver_lines_upper)
    ver_max = np.max(ver_lines_lower)
    hor_min = np.min(hor_lines_upper)
    hor_max = np.max(hor_lines_lower)
    framecenter_x = int((ver_max + ver_min) / 2)
    framecenter_y = int((hor_max + hor_min) / 2)
    square_scale = 0.5* (ver_max-ver_min) + 0.5*(hor_max - hor_min)

    return circles_f, framecenter_x, framecenter_y, square_scale

def getCropCoor(x,y,crop_size,image_width,image_height):
        ymin = y - crop_size
        ymax = y + crop_size

        xmin = x - crop_size
        xmax = x + crop_size

        if ymin<0:
            ymin = 0
            ymax = 2*crop_size
        if ymax>image_height-1:
            ymax = image_height-1
            ymin = image_height-1-2*crop_size
        if xmin<0:
            xmin = 0
            xmax = 2*crop_size
        if xmax>image_width-1:
            xmax = image_width-1
            xmin = image_width-1-2*crop_size
        return xmin,ymin,xmax,ymax


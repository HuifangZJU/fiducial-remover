from __future__ import division

import os.path

import matplotlib.pyplot as plt

from hough_utils import *
F_RADIUS = 15

def get_template_fiducial(path):
    img = plt.imread(path)
    img_g = img[:, :, 1]
    img_g = np.where(img_g < 10, img_g, 255 * np.ones(img_g.shape))
    img_g = Image.fromarray(img_g)
    img_g = img_g.convert('RGB')
    img_g = np.asarray(img_g)
    return img_g

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

def runCircle(fiducial_path,fiducial_radius=F_RADIUS):
    fiducials = get_template_fiducial(fiducial_path)

    circles_f = run_circle_threhold(fiducials, fiducial_radius, circle_threshold=30, step=2)
    # for i in range(circles_f.shape[0]):
    #     cv2.circle(fiducials, (circles_f[i, 0], circles_f[i, 1]), circles_f[i, 2], (0, 255, 0), 2)
    # plt.imshow(fiducials)
    # plt.show()
    return circles_f

def runSquare(fiducial_path):
    fiducials = get_template_fiducial(fiducial_path)
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






from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

F_RADIUS = 15

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





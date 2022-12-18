from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
path = '/home/huifang/workspace/data/mouse/posterior_v1/spatial/aligned_fiducials.jpg'
img = plt.imread(path)

img_r = img[:,:,0]
img_g = img[:,:,1]
img_b = img[:,:,2]

img_g = np.where(img_g<10, img_g, 255*np.ones(img_g.shape))
img_b = np.where(img_b<10, img_b, 255*np.ones(img_b.shape))
plt.imshow(img_b,cmap='gray',vmin=0, vmax=255)
plt.show()

fiducials = Image.fromarray(img_g)
fiducials = fiducials.convert('RGB')
fiducials.save('fiducials.jpeg')

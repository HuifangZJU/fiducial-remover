import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import morphology
from skimage.measure import label
# sample_image = 255*plt.imread('/home/huifang/workspace/data/fiducial_train/mouse/posterior_v2/spatial/tissue_hires_image.png')
# sample_image = sample_image.astype(np.uint8)
# plt.imshow(sample_image)
# plt.show()
# label_image = label(sample_image,neighbors=8, connectivity=3)
# plt.imshow(label_image)
# plt.show()
#
# img = morphology.remove_small_objects(label(sample_image),min_size=5,connectivity=1)
# plt.imshow(img)
# plt.show()

# img = cv2.imread('/home/huifang/workspace/data/fiducial_train/fiducial_9/spatial5/tissue_hires_image.png')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# f,a = plt.subplots(1,2)
# a[0].imshow(img)
# # img = cv2.bilateralFilter(img,15,80,80)
# img = label(img,neighbors=8)
# img_reserve = morphology.remove_small_objects(img,min_size=90)
#
# a[1].imshow(img_reserve)
# plt.show()
img = cv2.imread('/home/huifang/workspace/data/fiducial_train/humanpilot/151507/masks/human_in_loop_mask_solid_result.png')

#
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)

# plt.axis('off')
# plt.imshow(thresh)
# plt.show()

kernel = np.ones((3, 3), dtype=np.uint8)
edges = cv2.dilate(cv2.Canny(thresh,1200,1200),kernel,iterations=1)


# edges = cv2.Canny(thresh,1200,1200)
edges = label(edges,neighbors=4)
img_reserve = morphology.remove_small_objects(edges,min_size=2)
img_reserve = np.where(img_reserve>0,1,0)

plt.axis('off')
plt.imshow(img_reserve)
plt.show()
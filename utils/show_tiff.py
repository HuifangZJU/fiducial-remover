from matplotlib import pyplot as plt

img = plt.imread('./mouse/anterior_v1/img.tif')

img = img[7800:8150,670:1050,:]
plt.imshow(img)
plt.show()

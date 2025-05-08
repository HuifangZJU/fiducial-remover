from PIL import Image
from matplotlib import pyplot as plt
Image.MAX_IMAGE_PIXELS = None  # disables DecompressionBombError

# imglist = '/home/huifang/workspace/data/imagelists/tiff_img_list.txt'
# for i in [4,8]
imglist = "/home/huifang/workspace/data/imagelists/tiff_img_list.txt"
#[5,9,10,12]
annotation_path = '/media/huifang/data/fiducial/annotations/location_annotation/'
file = open(imglist)
lines = file.readlines()

# for i in range(len(lines)):
# for i in [5,9,10,12]:
# crop_boxed={"9":(23500, 15000, 25500,17000)}
for i in [3,4,12]:



    print(i)
    line = lines[i].rstrip().split(' ')
    # print(line)
    # test = input()
    #


    original_image = Image.open(line[0])

    smallimage = original_image.resize((2000,2000),Image.BICUBIC)
    plt.imshow(smallimage)
    plt.show()

    #
    # recovred_image = Image.open("/media/huifang/data/fiducial/tiff/V1_Human_Heart_image_recovered.tif")
    # crop_box = (1800,3500,3200,4500)  # adjust coordinates as needed
    # cropped_img = original_image.crop(crop_box)
    # cropped_img_recovered = recovred_image.crop(crop_box)
    # del original_image
    # del recovred_image
    #
    # # small_image = plt.imread(line[2])
    # # annotation_id =line[1]
    # # small_image = plt.imread(annotation_path + annotation_id + '.png')
    # f,a = plt.subplots(1,2)
    # a[0].imshow(cropped_img)
    # a[1].imshow(cropped_img_recovered)
    # plt.show()


    # img = Image.open('your_image.tiff')


# # Step 2: Crop a region (left, upper, right, lower)
# crop_box = (100, 100, 600, 600)  # adjust coordinates as needed
# cropped_img = img.crop(crop_box)
#
# # Step 3: Save as PNG
# cropped_img.save('cropped_image.png')



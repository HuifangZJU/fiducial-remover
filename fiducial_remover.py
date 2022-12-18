import argparse
import time

import imageio.plugins.feisem
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image

import torch
from matplotlib import pyplot as plt
from cnn_io import *
from fiducial_utils import read_tissue_image as read_image
from fiducial_utils import save_image as save


def get_to_be_recoved_patches(image,mask,circles,crop_size):
    circle_number = int(circles.shape[0])
    half = int(crop_size / 2)
    image_patches = []
    mask_patches = []
    location = []
    for i in range(circle_number):
        x = int(circles[i, 0])
        y = int(circles[i, 1])

        img_patch = image[y - half:y + half, x - half:x + half, :]
        img_patch = image_transformer(Image.fromarray(img_patch))
        img_patch = torch.unsqueeze(img_patch, dim=0)

        mask_patch = mask[y - half:y + half, x - half:x + half]
        mask_patch = mask_transformer(Image.fromarray(mask_patch))
        mask_patch = torch.unsqueeze(mask_patch, dim=0)

        image_patches.append(img_patch)
        mask_patches.append(mask_patch)
        location.append(circles[i, :])
    return image_patches,mask_patches,location

def pack_image_blocks(image_patches,mask_patches,location,ps,axis):
    #packing recovery blocks
    start_id=0
    image_blocks=[]
    mask_blocks=[]
    location_blocks=[]
    while start_id<len(image_patches):
        end_id = start_id+ps
        if end_id >= len(image_patches):
            end_id = len(image_patches)
        temp_img_block = image_patches[start_id]
        temp_mask_block = mask_patches[start_id]
        for id in range(start_id+1,end_id):
            try:
                temp_img_block = torch.cat((temp_img_block,image_patches[id]),axis)
                temp_mask_block = torch.cat((temp_mask_block, mask_patches[id]), axis)
            except:
                end_id=id
        temp_location_block = np.asarray(location[start_id:end_id])
        image_blocks.append(temp_img_block)
        mask_blocks.append(temp_mask_block)
        location_blocks.append(temp_location_block)
        start_id=end_id
    return image_blocks,mask_blocks,location_blocks


def run(img_np,mask_np,circles):


    image_patches,mask_patches,location = get_to_be_recoved_patches(img_np,mask_np,circles,args.img_height)


    image_blocks, mask_blocks, location_blocks = pack_image_blocks(image_patches, mask_patches, location,
                                                                   args.pack_size, 1)

    image_batches, mask_batches, location_batches = pack_image_blocks(image_blocks, mask_blocks, location_blocks,
                                                                   args.batch_size, 0)

    print('Packed to %s patches.'%len(image_batches))

    img_reserve = img_np.copy()
    i=0
    for image_block, mask_block, location_block in zip(image_batches, mask_batches, location_batches):
        print(str(len(image_patches))+'---'+str(i))
        i+=1

        img_var = Variable(image_block.type(Tensor))
        mask_var = Variable(mask_block.type(Tensor))
        t1 = time.time()
        ps = int(image_block.shape[1] / args.channel)
        bs = image_block.shape[0]
        recover_var = getReconstructedImg(bs,ps,args.channel, img_var, mask_var, device,
                                          Tensor, 600)

        t2 = time.time()
        # print('Single run cost %.2f seconds.' % (t2 - t1))
        # print(recover_var.shape)
        # test = input()

        recover_np = torch_to_np(recover_var)
        recover_np = np.transpose(recover_np, (0, 2, 3, 1))
        image_block = np.transpose(image_block, (0, 2, 3, 1))
        mask_block = np.transpose(mask_block, (0, 2, 3, 1))
        half = int(args.img_height/2)
        for batch_id in range(bs):
            recover_np_temp = recover_np[batch_id,:]
            ps = int(recover_np_temp.shape[-1]/args.channel)
            for pack_id in range(ps):
                x = int(location_block[batch_id,pack_id,0])
                y = int(location_block[batch_id,pack_id,1])
                sch = pack_id*args.channel
                ech = (pack_id+1)*args.channel
                img_reserve[y-half:y+half,x-half:x+half,:] = recover_np_temp[:,:,sch:ech]*255

        # f,a = plt.subplots(3,4)
        # a[0, 0].imshow(image_block[0, :,:,:3])
        # a[0, 1].imshow(image_block[0, :,:,3:6])
        # a[0, 2].imshow(image_block[0, :,:,6:9])
        # a[0, 3].imshow(image_block[0, :,:,9:12])
        # a[1, 0].imshow(mask_block[0, :,:,0])
        # a[1, 1].imshow(mask_block[0, :,:,1])
        # a[1, 2].imshow(mask_block[0, :,:,2])
        # a[1, 3].imshow(mask_block[0, :,:,3])
        # a[2, 0].imshow(recover_np[0, :,:,:3])
        # a[2, 1].imshow(recover_np[0, :,:,3:6])
        # a[2, 2].imshow(recover_np[0, :,:,6:9])
        # a[2, 3].imshow(recover_np[0, :,:,9:12])
        #
        # plt.show()



    return img_reserve

# ------ arguments handling -------
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--pack_size', type=int, default=4, help='size of deep a image training patches')
parser.add_argument('--img_height', type=int, default=32, help='size of image height')
parser.add_argument('--img_width', type=int, default=32, help='size of image width')
parser.add_argument('--channel', type=int, default=3, help='number of image channel')
args = parser.parse_args()
os.makedirs('./test/', exist_ok=True)

# ------ device handling -------
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------ Configure model -------
# Initialize generator
generator = getGenerator()
generator.to(device)

# ------ main process -------
# manage input
# transformer = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
#                transforms.ToTensor(),
#                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transformer = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
               transforms.ToTensor()]
transformer2 = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
               transforms.ToTensor()]

image_transformer=transforms.Compose(transformer)
mask_transformer = transforms.Compose(transformer2)



root_path = '/home/huifang/workspace/data/fiducial_train/'
masktype='hough_mask_solid_r2'
dataset_names = os.listdir(root_path)
for dataset_name in dataset_names:
    print('-------------'+dataset_name+'-------------')
    image_names= os.listdir(root_path+dataset_name)
    for image_name in image_names:
        print(image_name+'...')
        img_np = 255*read_image(root_path+dataset_name+'/'+ image_name)
        mask_np = 255*plt.imread(root_path+dataset_name+'/'+ image_name +'/masks/'+ masktype + '.png')
        mask_np = 255 - mask_np
        circles = np.load(root_path+dataset_name+'/'+ image_name+'/masks/'+ masktype + '.npy')

        img_np = img_np.astype(np.uint8)
        mask_np = mask_np.astype(np.uint8)
        t1 = time.time()
        recovered_image = run(img_np, mask_np, circles)
        t2 = time.time()
        print('Recover cost %.2f seconds.' % (t2-t1))
        save(recovered_image,root_path+dataset_name+'/'+ image_name+'/masks/'+ masktype + '_result.png')
        print('current image done')
        # plt.imshow(recovered_image)
        # plt.show()
    print('current data set done')



# img = image_transformer(img)
# img = torch.unsqueeze(img,dim=0)
# img_var = Variable(img.type(Tensor))

#------from model------#
# mask_var = generator(img_var)
# recover_var = getReconstructedImg(args.batch_size,args.channel,img_var,mask_var,img_np.shape[:2],device,Tensor,400)
#------from file------#





# test_sample= torch.cat((img_var.data, recover_var.data), -2)
# save_image(test_sample, './test/%s_img.png' % i, normalize=True)
# save_image(mask_var.data, './test/%s_mask.png' % i, normalize=True)
# print('done')


# #test single image
# img = Image.open('../../../data/humanpilot/151507/spatial/tissue_hires_image.png')
# img_np = np.array(img)
# image_transformer=transforms.Compose(transformer)
# img = image_transformer(img)
# img = torch.unsqueeze(img,dim=0)
# img_tensor = Variable(img.type(Tensor))
# mask = generator(img_tensor)
#
# image_transformer2=transforms.Compose(transformer2)
# mask = image_transformer2(mask)
# # save_image(output.data, './test/151507_pix2.png', normalize=True)
# mask = torch.squeeze(mask)
# mask = mask.cpu().detach().numpy()
#
#
# img_a = torch.squeeze(img)
# img_a = img_a.numpy()
# index = np.where(mask > -0.5)
# x = index[0]
# y = index[1]
# max = img_a.max()
#
# for i,j in zip(x,y):
#     img_a[:,i,j]=[max,max,max]
#
# img_a = np.transpose(img_a, (1, 2, 0))
# plt.imshow(img_a)
# plt.show()

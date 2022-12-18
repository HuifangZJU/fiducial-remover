from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim

from utils.inpainting_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize = -1
dim_div_by = 32

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'

img_path  = 'data/inpainting/1_image.png'
mask_path = 'data/inpainting/1_mask.png'


NET_TYPE = 'UNET' # one of skip_depth4|skip_depth2|UNET|ResNet

img_pil, _ = get_image(img_path, (32,32))
img_mask_pil, _ = get_image(mask_path, (32,32))
img_mask_pil = crop_image(img_mask_pil, dim_div_by)
img_pil= crop_image(img_pil,dim_div_by)

img_np= pil_to_np(img_pil)
img_mask_np= pil_to_np(img_mask_pil)


img_mask_var = np_to_torch(img_mask_np).type(dtype)

f, a = plt.subplots(1, 2)
img_temp = np.transpose(img_np, (1, 2, 0))
a[0].imshow(img_temp)
mask_temp = np.transpose(img_mask_np, (1, 2, 0))
a[1].imshow(mask_temp)
plt.show()


pad = 'reflection' # 'zero'
OPT_OVER = 'net'
OPTIMIZER = 'adam'

INPUT = 'noise'
input_depth = 1

num_iter = 3001
show_every = 50
figsize = 8
reg_noise_std = 0.00
param_noise = True

if 'skip' in NET_TYPE:

    depth = int(NET_TYPE[-1])
    net = skip(input_depth, img_np.shape[0],
               num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
               num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
               num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
               filter_size_up=3, filter_size_down=5, filter_skip_size=1,
               upsample_mode='nearest',  # downsample_mode='avg',
               need1x1_up=False,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    LR = 0.01

elif NET_TYPE == 'UNET':

    net = UNet(num_input_channels=input_depth, num_output_channels=3,
               feature_scale=8, more_layers=0,
               concat_x=False, upsample_mode='deconv',
               pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

    LR = 0.001
    param_noise = False

elif NET_TYPE == 'ResNet':

    net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU')

    LR = 0.001
    param_noise = False


net = net.type(dtype)
net_input = get_noise(1,input_depth, INPUT, img_np.shape[1:]).type(dtype)
# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(img_mask_np).type(dtype)
img_np = np.transpose(img_np, (1, 2, 0))
i = 0


def closure():
    global i

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    net.to(device)
    net_input.to(device)

    out = net(net_input)

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
    if PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        f,a = plt.subplots(1,2)
        print(i)
        a[0].imshow(img_np)
        temp = np.clip(out_np, 0, 1)
        temp= np.transpose(temp, (1, 2, 0))
        a[1].imshow(temp)
        plt.show()
        # plot_image_grid([img_np,np.clip(out_np, 0, 1)], factor=figsize, nrow=2)

    i += 1

    return total_loss


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_np = torch_to_np(net(net_input))
plot_image_grid([out_np], factor=5)



















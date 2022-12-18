from pix2pix.models import *
from pix2pix.datasets import *
import numpy as np
from dip.models.unet import UNet_Parrel
from dip.models.skip import skip
import torch
import torch.optim
from dip.utils.inpainting_utils import *
import seaborn as sns

BASE_PATH = '/home/huifang/workspace/'
def getGenerator():
    generator = Generator()
    generator.load_state_dict(torch.load(BASE_PATH + 'experiment/pix2pix/saved_models/initial-pixel-loss/g_400.pth'))
    return generator


def getInpainter(input_channel,output_channel,packsize):

    # net = UNet(num_input_channels=input_channel, num_output_channels=output_channel,
    #            feature_scale=2, more_layers=0,
    #            concat_x=False, upsample_mode='deconv',
    #            pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

    net = UNet_Parrel(packsize,num_input_channels=input_channel, num_output_channels=output_channel,
               feature_scale=2, more_layers=0,
               concat_x=False, upsample_mode='deconv',
               pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

    lr = 0.001
    return net, lr


def getReconstructedImg(batch_size,pack_size,input_channel,img_var, mask_var, device,Tensor,num_iter=400):
    input_depth = 3*pack_size

    net, LR = getInpainter(input_channel, input_channel,pack_size)

    net.to(device)
    INPUT = 'noise'
    net_input = get_noise(batch_size,input_depth, INPUT, img_var.shape[-2:])
    net_input = Variable(net_input.type(Tensor))
    # Loss
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    show_every=num_iter

    for j in range(1,num_iter+1):
        optimizer.zero_grad()
        out = net(net_input)
        loss = mse(out[:,:input_channel,:,:] * mask_var[:,:1,:,:], img_var[:,:input_channel,:,:] * mask_var[:,:1,:,:])
        for p in range(1,pack_size):
            sch = p*input_channel
            ech = (p+1)*input_channel
            loss += mse(out[:,sch:ech, :, :] * mask_var[:, p:p+1, :, :],
                       img_var[:, sch:ech, :, :] * mask_var[:, p:p+1, :, :])

        loss.backward()
        optimizer.step()
        # print('Iteration %05d ' % j, '\r', end='')
        # if j % show_every == 0:
        #     out_np = torch_to_np(out)
        #     img_np = torch_to_np(img_var)
        #     img_np = np.transpose(img_np, (1, 2, 0))
        #     f, a = plt.subplots(2, 2)
        #     print(j)
        #     a[0,0].imshow(img_np)
        #     mask_np = torch_to_np(mask_var)
        #     mask_np = np.transpose(mask_np, (1, 2, 0))
        #
        #     a[0,1].imshow(mask_np,cmap='gray')
        #     a[1, 0].imshow(img_np*mask_np)
        #     temp = np.clip(out_np, 0, 1)
        #     temp = np.transpose(temp, (1, 2, 0))
        #     a[1,1].imshow(temp)


            # img_r = img_np[:,:,0]
            # img_g = img_np[:,:,1]
            # img_b = img_np[:,:,2]
            # img_r.sort()
            # img_g.sort()
            # img_b.sort()
            #
            # sns.displot(img_r, bins=30)
            # sns.displot(img_g, bins=30)
            # sns.displot(img_b, bins=30)
            # plt.show()
    # print("done")

    recover_var = net(net_input)
    return recover_var

















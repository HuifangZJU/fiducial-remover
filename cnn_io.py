import matplotlib.pyplot as plt

from pix2pix.models import *
from pix2pix.datasets import *
import numpy as np
from dip.models.unet import UNet
from dip.models.skip import skip
import torch
import torch.optim
from dip.utils.inpainting_utils import *
import seaborn as sns
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule

BASE_PATH = '/home/huifang/workspace/'
def get_circle_Generator():
    generator = Generator()
    generator.load_state_dict(torch.load('/media/huifang/data/experiment/pix2pix/saved_models/final_circle_width2_unet_with_aug_mse/g_600.pth'))
    # generator.load_state_dict(torch.load('/media/huifang/data/experiment/pix2pix/saved_models/width2_downsample_nocondition_lamda10_with_0.125negative/g_400.pth'))

    # generator = Dot_Generator()
    # generator.load_state_dict(torch.load(
    #     '/media/huifang/data/experiment/pix2pix/saved_models/transformer-dot-sigmoid-mse-5pe-noskip/g_800.pth'))
    return generator
def get_position_Generator():
    generator = Patch_Binary_Generator()
    generator.load_state_dict(torch.load('/media/huifang/data/experiment/pix2pix/saved_models/binary-square-alltrain-5-pe/g_400.pth'))
    # generator.load_state_dict(
    #     torch.load('/media/huifang/data/experiment/pix2pix/saved_models/binary-square-5pe-with-aug-nocrop/g_800.pth'))
    return generator

def get_combined_Generator():
    # generator = CNN_in_parrel_Generator()
    # generator.load_state_dict((torch.load(
    #     '/media/huifang/data/experiment/pix2pix/saved_models/auto_pure_cnn_0layer_with_binary_with_spatial/g_1400.pth')))
    # generator = Attention_Generator()
    # generator.load_state_dict((torch.load(
    #     '/media/huifang/data/experiment/pix2pix/saved_models/auto_attn_net_5layer/g_800.pth')))


    generator = Rich_Parrel_Attention_Generator()
    # generator.load_state_dict((torch.load(
    #     '/media/huifang/data/experiment/pix2pix/saved_models/auto_annotation_circle_only/g_600.pth')))
    generator.load_state_dict((torch.load(
        '/media/huifang/data/experiment/pix2pix/saved_models/auto_circle_binary_spatial_selected_mask_5layer_renewed/g_1600.pth')))
    # generator.load_state_dict((torch.load(
    #     '/media/huifang/data/experiment/pix2pix/saved_models/ground_truth_circle_only/g_600.pth')))

    # generator.load_state_dict((torch.load(
    #     '/media/huifang/data/experiment/pix2pix/saved_models/auto_circle_binary_selected_mask_5layer/g_800.pth')))


    return generator

def getInpainter(input_channel,output_channel):

    net = UNet(num_input_channels=input_channel, num_output_channels=output_channel,
               feature_scale=2, more_layers=0,
               concat_x=False, upsample_mode='deconv',
               pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

    # net = UNet_Parrel(packsize,num_input_channels=input_channel, num_output_channels=output_channel,
    #            feature_scale=2, more_layers=0,
    #            concat_x=False, upsample_mode='deconv',
    #            pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

    lr = 0.001
    return net, lr


# def getParrelReconstructedImg(batch_size,pack_size,input_channel,img_var, mask_var, device,Tensor,num_iter=400):
#     input_depth = 3*pack_size
#
#     net, LR = getInpainter(input_channel, input_channel,pack_size)
#
#     net.to(device)
#     INPUT = 'noise'
#     net_input = get_noise(batch_size,input_depth, INPUT, img_var.shape[-2:])
#     net_input = net_input.type(Tensor)
#     # Loss
#     mse = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(net.parameters(), lr=LR)
#     show_every=num_iter
#
#     for j in range(1,num_iter+1):
#         optimizer.zero_grad()
#         out = net(net_input)
#         loss = mse(out[:,:input_channel,:,:] * mask_var[:,:1,:,:], img_var[:,:input_channel,:,:] * mask_var[:,:1,:,:])
#         for p in range(1,pack_size):
#             sch = p*input_channel
#             ech = (p+1)*input_channel
#             loss += mse(out[:,sch:ech, :, :] * mask_var[:, p:p+1, :, :],
#                        img_var[:, sch:ech, :, :] * mask_var[:, p:p+1, :, :])
#
#         loss.backward()
#         optimizer.step()
#         # print('Iteration %05d ' % j, '\r', end='')
#         # if j % show_every == 0:
#         #     out_np = torch_to_np(out)
#         #     img_np = torch_to_np(img_var)
#         #     img_np = np.transpose(img_np, (1, 2, 0))
#         #     f, a = plt.subplots(2, 2)
#         #     print(j)
#         #     a[0,0].imshow(img_np)
#         #     mask_np = torch_to_np(mask_var)
#         #     mask_np = np.transpose(mask_np, (1, 2, 0))
#         #
#         #     a[0,1].imshow(mask_np,cmap='gray')
#         #     a[1, 0].imshow(img_np*mask_np)
#         #     temp = np.clip(out_np, 0, 1)
#         #     temp = np.transpose(temp, (1, 2, 0))
#         #     a[1,1].imshow(temp)
#
#     # print("done")
#
#     recover_var = net(net_input)
#     return recover_var



def getReconstructedImg(img_var, mask_var, device,Tensor,num_iter=400):
    input_depth = 3

    net, LR = getInpainter(3, 3)

    net.to(device)
    INPUT = 'noise'
    net_input = get_noise(input_depth, INPUT, img_var.shape[-2:])
    net_input = net_input.type(Tensor)
    # Loss
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    show_every=50

    for j in range(1,num_iter+1):
        optimizer.zero_grad()
        out = net(net_input)
        loss = mse(out* mask_var, img_var * mask_var)
        loss.backward()
        optimizer.step()
        print('Iteration %05d ' % j, '\r', end='')
        if j % show_every == 0:
            out_np = torch_to_np(out.squeeze())
            img_np = torch_to_np(img_var.squeeze())
            img_np = np.transpose(img_np, (1, 2, 0))
            f, a = plt.subplots(2, 2)
            print(j)
            a[0,0].imshow(img_np)
            mask_np = torch_to_np(mask_var.squeeze())
            # mask_np = np.transpose(mask_np, (1, 2, 0))

            a[0,1].imshow(mask_np,cmap='gray')
            a[1, 0].imshow(img_np*mask_np[:, :, None])
            temp = np.clip(out_np, 0, 1)
            temp = np.transpose(temp, (1, 2, 0))
            a[1,1].imshow(temp)
            plt.show()

    # print("done")

    recover_var = net(net_input)
    return recover_var


def getLamaInpainter(train_config, path, map_location='cuda', strict=True):
    kwargs = dict(train_config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = train_config.trainer.kwargs.get('accelerator', None) == 'ddp'
    model = DefaultInpaintingTrainingModule(train_config, **kwargs)

    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state['state_dict'], strict=strict)
    model.on_load_checkpoint(state)
    return model










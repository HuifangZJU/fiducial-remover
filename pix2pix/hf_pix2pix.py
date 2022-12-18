import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from itertools import cycle
import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=600, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=801, help='number of epochs of training')
parser.add_argument('--pretrained_name', type=str, default="new_width2_downsample_finetune_onlypixel",
                    help='name of the dataset')
parser.add_argument('--model_dir', type=str, default="width2_tissue_human_onlypixel", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=2048, help='size of image height')
parser.add_argument('--img_width', type=int, default=2048, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50,
                    help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between model checkpoints')
args = parser.parse_args()


experiment_path = '/home/huifang/workspace/experiment/pix2pix'
image_save_path = experiment_path + '/images'
model_save_path = experiment_path + '/saved_models'
log_save_path = experiment_path + '/logs'
os.makedirs(image_save_path + '/%s' % args.model_dir, exist_ok=True)
os.makedirs(model_save_path + '/%s' % args.model_dir, exist_ok=True)
os.makedirs(log_save_path + '/%s' % args.model_dir, exist_ok=True)

train_data_list ="/home/huifang/workspace/data/imagelists/fiducial_tissue_human.txt"
test_data_list = "/home/huifang/workspace/data/imagelists/crop_image_hard.txt"

# ------------------------------------------
#                Training preparation
# ------------------------------------------
# ------ device handling -------
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'

# ------ Configure loss -------
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 20

# ------ Configure model -------
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
if args.epoch != 0:
    generator.load_state_dict(torch.load(model_save_path +'/%s/g_%d.pth' % (args.pretrained_name, args.epoch)))
    discriminator.load_state_dict(torch.load(model_save_path +'/%s/d_%d.pth' % (args.pretrained_name, args.epoch)))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
generator.to(device)
discriminator.to(device)
# Calculate output of image discriminator (PatchGAN)
patch = (1, args.img_height // 2 ** 2, args.img_width // 2 ** 2)

# ------ Configure optimizer -------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# ------ Configure data loaders -------
# Configure dataloaders
transforms_rgb = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transforms_gray = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5), (0.5))]


train_dataloader = DataLoader(ImageDataset(train_data_list, transforms_a=transforms_rgb,transforms_b=transforms_gray),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
test_dataloader = DataLoader(ImageTestDataset(test_data_list, transforms_=transforms_rgb, ),
                             batch_size=1, shuffle=True, num_workers=args.n_cpu)
test_samples = cycle(test_dataloader)
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(epoch,batches_done):
    """Saves a generated sample from the validation set"""
    test_batch = next(test_samples)
    test_a = test_batch['A']
    real_a = Variable(test_a.type(Tensor))
    output = generator(real_a)
    # img_sample = torch.cat((test_a.data, output.data), -2)
    save_image(test_a.data, image_save_path+'/%s/%s_%s_img.png' % (args.model_dir,epoch,batches_done), nrow=4, normalize=True)
    save_image(output.data, image_save_path+'/%s/%s_%s_mask.png' % (args.model_dir,epoch, batches_done), nrow=4, normalize=True)


# ------------------------------------------
#                Training
# ------------------------------------------
prev_time = time.time()
logger = SummaryWriter(log_save_path)

for epoch in range(args.epoch, args.n_epochs):
    for i, batch in enumerate(train_dataloader):
        real_A = batch['A']
        real_B = batch['B']

        valid = Variable(Tensor(np.ones((real_B.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_B.size(0), *patch))), requires_grad=False)

        # Model inputs
        real_A = Variable(real_A.type(Tensor))
        real_B = Variable(real_B.type(Tensor))
        fake_B = generator(real_A)

        optimizer_G.zero_grad()
        # GAN loss
        pred_fake = discriminator(fake_B, real_A)

        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = lambda_pixel * criterion_pixelwise(fake_B, real_B)

        # Total loss
        # loss_G = loss_GAN + loss_pixel
        loss_G = loss_pixel
        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = args.n_epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r" + args.model_dir + "---[Epoch %d/%d] [Batch %d/%d] [Loss G: %f] [Loss D: %f] ---[Loss GAN: %f  Loss pixel: %f]  ETA: %s" %
            (epoch, args.n_epochs,
             i, len(train_dataloader),
             loss_G.item(), loss_D.item(), loss_GAN.item(),loss_pixel.item(), time_left))
        # # If at sample interval save image
        # if batches_done % args.sample_interval == 0:
        #     sample_images(epoch, batches_done)
        # --------------tensor board--------------------------------#
        # if batches_done % 100 == 0:
        #     info = {'loss_G': loss_G.item(), 'loss_D': loss_D.item()}
        #     for tag, value in info.items():
        #         logger.add_scalar(tag, value, batches_done)
        #     for tag, value in generator.named_parameters():
        #         tag = tag.replace('.', '/')
        #         logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)
        #         # logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)
        #     for tag, value in discriminator.named_parameters():
        #         tag = tag.replace('.', '/')
        #         logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)
        #         # logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), model_save_path+'/%s/g_%d.pth' % (args.model_dir,epoch))
        torch.save(discriminator.state_dict(), model_save_path+'/%s/d_%d.pth' % (args.model_dir,epoch))

# save final model
torch.save(generator.state_dict(),  model_save_path+'/%s/g_%d.pth' % (args.model_dir,epoch))
torch.save(discriminator.state_dict(),  model_save_path+'/%s/d_%d.pth' % (args.model_dir,epoch))
logger.close()

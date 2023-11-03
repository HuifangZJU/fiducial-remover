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
from utils import divide_batch_into_patches, reconstruct_batch_images


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=401, help='number of epochs of training')
parser.add_argument('--pretrained_name', type=str, default="",
                    help='name of the dataset')
parser.add_argument('--model_dir', type=str, default="final_attn_unet_test", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50,
                    help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between model checkpoints')
args = parser.parse_args()


experiment_path = '/media/huifang/data/experiment/pix2pix'
image_save_path = experiment_path + '/images'
model_save_path = experiment_path + '/saved_models'
log_save_path = experiment_path + '/logs'
os.makedirs(image_save_path + '/%s' % args.model_dir, exist_ok=True)
os.makedirs(model_save_path + '/%s' % args.model_dir, exist_ok=True)
os.makedirs(log_save_path + '/%s' % args.model_dir, exist_ok=True)

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
criterion_binary = torch.nn.MSELoss()
# ------ Configure model -------
# Initialize generator
generator = Attention_Generator(with_skip_connection=False)
if args.epoch != 0:
    generator.load_state_dict(torch.load(model_save_path +'/%s/g_%d.pth' % (args.pretrained_name, args.epoch)))
else:
    generator.apply(weights_init_normal)
generator.to(device)
# ------ Configure optimizer -------
optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# ------ Configure data loaders -------
# Configure dataloaders
transforms_rgb = [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transforms_gray = [transforms.ToTensor()]


test_data_set = AttnDataset(transforms_a=transforms_rgb,transforms_b=transforms_gray,mode='test',test_group=1,aug=False)
print(len(test_data_set))
for i in range(test_data_set.__len__(),0,-1):
    print(i)
    batch= test_data_set.__getitem__(i)
    f,a = plt.subplots(1,2)
    a[0].imshow(batch['A'])
    a[1].imshow(batch['B'])
    plt.show()

train_dataloader = DataLoader(AttnDataset(transforms_a=transforms_rgb,transforms_b=transforms_gray,mode='train',test_group=1,aug=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
test_dataloader = DataLoader(AttnDataset(transforms_a=transforms_rgb,transforms_b=transforms_gray,mode='test',test_group=1,aug=False),
                             batch_size=1, shuffle=False, num_workers=args.n_cpu)

test_samples = cycle(test_dataloader)
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(epoch,batches_done):
    """Saves a generated sample from the validation set"""
    test_batch = next(test_samples)
    test_image = test_batch['A'].to(device)
    test_labels = test_batch['B'].to(device)
    output = generator(test_image)
    # img_sample = torch.cat((test_a.data, output.data), -2)
    save_image(test_image.data, image_save_path+'/%s/%s_%s_img.png' % (args.model_dir,epoch,batches_done), nrow=4, normalize=True)
    save_image(test_labels.data, image_save_path+'/%s/%s_%s_gt.png' % (args.model_dir,epoch, batches_done), nrow=4, normalize=True)
    save_image(output.data, image_save_path + '/%s/%s_%s_mask.png' % (args.model_dir,epoch, batches_done), nrow=4, normalize=True)


# ------------------------------------------
#                Training
# ------------------------------------------
prev_time = time.time()
logger = SummaryWriter(log_save_path + '/%s' % args.model_dir)

for epoch in range(args.epoch, args.n_epochs):
    for i, batch in enumerate(train_dataloader):
        images = batch['A'].to(device)
        labels = batch['B'].to(device)
        predictions = generator(images)
        optimizer.zero_grad()
        # compute loss
        loss = criterion_binary(predictions,labels)
        # loss_G = loss_pixel
        loss.backward()

        optimizer.step()

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
            "\r" + args.model_dir + "---[Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s" %
            (epoch, args.n_epochs,
             i, len(train_dataloader),
             loss.item(), time_left))
        # # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(epoch, batches_done)
        # --------------tensor board--------------------------------#
        if batches_done % 100 == 0:
            info = {'loss': loss.item()}
            for tag, value in info.items():
                logger.add_scalar(tag, value, batches_done)
            for tag, value in generator.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)
                # logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        torch.save(generator.state_dict(), model_save_path+'/%s/g_%d.pth' % (args.model_dir,epoch))

# save final model
torch.save(generator.state_dict(),  model_save_path+'/%s/g_%d.pth' % (args.model_dir,epoch))
logger.close()

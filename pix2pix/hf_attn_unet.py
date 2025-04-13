import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from losses import *

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
def get_image_mask_from_annotation(image_size,annotation,step):
    image_mask = np.zeros(image_size)

    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            patch_x = i * step
            patch_y = j * step
            image_mask[patch_x:patch_x + step, patch_y:patch_y + step] = annotation[i, j]
    return image_mask


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=801, help='number of epochs of training')
parser.add_argument('--pretrained_name', type=str, default="",
                    help='name of the dataset')
parser.add_argument('--model_dir', type=str, default="final_attn_test", help='name of the dataset')
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
criterion = FocalLoss(alpha=0.95)
# ------ Configure model -------
# Initialize generator
generator = Attention_Generator()
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

train_dataloader = DataLoader(AttnInparrelDataset(transforms_a=transforms_rgb,transforms_b=transforms_gray,mode='train',test_group=1),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
test_dataloader = DataLoader(AttnInparrelDataset(transforms_a=transforms_rgb,transforms_b=transforms_gray,mode='test',test_group=1),
                             batch_size=1, shuffle=False, num_workers=args.n_cpu)



# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
test_samples = cycle(test_dataloader)

def calculate_iou_dice(ground_truth, prediction, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) and Dice Coefficient for binary segmentation.

    :param ground_truth: PyTorch tensor of ground truth, shape [1, 1, m, n]
    :param prediction: PyTorch tensor of predictions (sigmoid output), shape [1, 1, m, n]
    :param threshold: Threshold to convert sigmoid output to binary format
    :return: IoU and Dice Coefficient scores
    """
    # Flatten the tensors
    ground_truth_flat = ground_truth.view(-1).bool()
    prediction_flat = prediction.view(-1) >= threshold  # Apply threshold

    # Intersection and union
    intersection = (ground_truth_flat & prediction_flat).float().sum()
    union = (ground_truth_flat | prediction_flat).float().sum()

    # Calculate IoU and Dice
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (ground_truth_flat.float().sum() + prediction_flat.float().sum() + 1e-6)

    return iou.item(), dice.item()

def compute_test_accuracy():
    """Saves a generated sample from the validation set"""
    test_batch = next(test_samples)
    test_image = test_batch['A'].to(device)
    test_gt = test_batch['B'].to(device)
    unet_output = generator(test_image)
    iou, dice = calculate_iou_dice(test_gt, unet_output)
    save_image(test_image.data, image_save_path+'/%s/%s_%s_img.png' % (args.model_dir,epoch,batches_done), nrow=4, normalize=True)
    save_image(unet_output.data, image_save_path + '/%s/%s_%s_mask.png' % (args.model_dir,epoch, batches_done), nrow=4, normalize=True)
    return iou,dice
# ------------------------------------------
#                Training
# ------------------------------------------
prev_time = time.time()
logger = SummaryWriter(log_save_path + '/%s' % args.model_dir)

for epoch in range(args.epoch, args.n_epochs):
    for i, batch in enumerate(train_dataloader):
        images = batch['A'].to(device)
        masks = batch['B'].to(device)
        unet_output = generator(images)
        optimizer.zero_grad()
        # compute loss
        loss = criterion(unet_output , masks)
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
        # --------------tensor board--------------------------------#
        if batches_done % args.sample_interval == 0:
            iou, dice = compute_test_accuracy()
            info = {'loss': loss.item(), 'test_iou':iou}
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

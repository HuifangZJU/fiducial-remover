import sys
import argparse
import time
import datetime
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from losses import FocalLoss
from models import *
from datasets import *
from itertools import cycle
import torch
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=601, help='number of epochs of training')
parser.add_argument('--pretrained_name', type=str, default="",
                    help='name of the dataset')
parser.add_argument('--model_dir', type=str, default="final_circle_full_image_test", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=32, help='size of image height')
parser.add_argument('--img_width', type=int, default=32, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50,
                    help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between model checkpoints')
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
# criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCELoss()
criterion = FocalLoss(alpha=0.95)
# ------ Configure model -------
# Initialize generator and discriminator
generator = Generator()
if args.epoch != 0:
    generator.load_state_dict(torch.load(model_save_path +'/%s/g_%d.pth' % (args.pretrained_name, args.epoch)))
else:
    generator.apply(weights_init_normal)

generator.to(device)
# ------ Configure optimizer -------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
# ------ Configure data loaders -------
# Configure dataloaders
# transforms_rgb = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
#                 transforms.ToTensor(),
#                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# transforms_gray = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
#             transforms.ToTensor()]

transforms_rgb = [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transforms_gray = [transforms.ToTensor()]

transforms_test = [transforms.Resize((2048, 2048), Image.BICUBIC),
                transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


image_list ="/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt"
circle_list = "/home/huifang/workspace/data/imagelists/st_trainable_circles_downsample_with_negative_final.txt"

# test_data_set = CircleTrainFullImageDataset(transforms_a=transforms_rgb,transforms_b=transforms_gray,test_group=1)
# for i in range(0,test_data_set.__len__()):
#     print(i)
#     batch= test_data_set.__getitem__(i)
#     f,a = plt.subplots(1,2)
#     a[0].imshow(batch['A'])
#     a[1].imshow(batch['B'])
#     plt.show()
# train_dataloader = DataLoader(CircleTrainDataset(circle_list,transforms_a=transforms_rgb,transforms_b=transforms_gray,test_group=1),
#                               batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

train_dataloader = DataLoader(CircleTrainFullImageDataset(transforms_a=transforms_rgb,transforms_b=transforms_gray,test_group=1),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
test_dataloader = DataLoader(CircleTestDataset(image_list, transforms_=transforms_rgb, test_group=1),
                             batch_size=1, shuffle=False, num_workers=args.n_cpu)
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
logger = SummaryWriter(log_save_path + '/%s' % args.model_dir)

for epoch in range(args.epoch, args.n_epochs):
    for i, batch in enumerate(train_dataloader):
        real_A = batch['A']
        real_B = batch['B']

        # Model inputs
        real_A = Variable(real_A.type(Tensor))
        real_B = Variable(real_B.type(Tensor))
        fake_B = generator(real_A)

        optimizer_G.zero_grad()
        # Pixel-wise loss
        loss_pixel = criterion(fake_B, real_B)
        loss_pixel.backward()
        optimizer_G.step()

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
            "\r" + args.model_dir + "---[Epoch %d/%d] [Batch %d/%d] [Loss pixel: %f]  ETA: %s" %
            (epoch, args.n_epochs,
             i, len(train_dataloader),
             loss_pixel.item(), time_left))
        # # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(epoch, batches_done)
        # --------------tensor board--------------------------------#
        if batches_done % 100 == 0:
            info = {'loss_pixel': loss_pixel.item()}
            for tag, value in info.items():
                logger.add_scalar(tag, value, batches_done)
            for tag, value in generator.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), model_save_path+'/%s/g_%d.pth' % (args.model_dir,epoch))

# save final model
torch.save(generator.state_dict(),  model_save_path+'/%s/g_%d.pth' % (args.model_dir,epoch))
logger.close()

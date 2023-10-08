import argparse
from PIL import ImageFilter

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import *
from datasets import *
import torch
from matplotlib import pyplot as plt



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
parser.add_argument('--model_name', type=str, default="transformer",
                    help='name of the dataset')
# parser.add_argument('--image_path', type=str, default='/media/huifang/data/fiducial/original_data/10x/CytAssist/CytAssist_11mm_FFPE_Human_Colorectal_Cancer_spatial/spatial/cytassist_image.tiff', help='path to image')
parser.add_argument('--image_path', type=str, default='/media/huifang/data/fiducial/data/12_STDS0000119_Brain_SR_map/2/tissue_hires_image.png', help='path to image')

parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--img_height', type=int, default=3600, help='size of image height')
parser.add_argument('--img_width', type=int, default=3600, help='size of image width')
args = parser.parse_args()

save_dir = './test/%s/' % args.model_name
os.makedirs('./test/%s' % args.model_name, exist_ok=True)
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
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# ------ Configure model -------
# Initialize generator
generator = Attention_Generator()
# generator = Attention_Generator()
BASE_PATH = '/media/huifang/data/'
generator.load_state_dict(torch.load(BASE_PATH + 'experiment/pix2pix/saved_models/%s/g_%d.pth' % (args.model_name, args.epoch)))
generator.to(device)

# ------ Configure data loaders -------
# Configure dataloaders

transformer = [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

test_dataloader = DataLoader(BinaryDataset(transforms_=transformer),
                             batch_size=1, shuffle=False, num_workers=args.n_cpu)
test_samples = iter(test_dataloader)
#
#
# for i, test_batch in enumerate(test_dataloader):
#     test_a = test_batch['A']
#     real_a = Variable(test_a.type(Tensor))
#     output = generator(real_a)
#     # img_sample = torch.cat((test_a.data, output.data), -2)
#     save_image(test_a.data, './test/%s/%s_img.png' % (args.model_name, i), nrow=4, normalize=True)
#     save_image(output.data, './test/%s/%s_mask.png' % (args.model_name, i), nrow=4, normalize=True)

img_a = Image.open(args.image_path)
#img_a = Image.open('/home/huifang/workspace/data/fiducial_eval/eval/spatial6/tissue_hires_image.png')
#img_a = Image.open('/home/huifang/workspace/data/fiducial_train/humanpilot/151507/spatial/tissue_hires_image.png')
# img_a = Image.open('/home/huifang/workspace/data/fiducial_eval/eval/spatial7/tissue_hires_image.png')
img_np = np.asarray(img_a)
# img_a = Image.open('../../../data/humanpilot/151509/spatial/tissue_hires_image.png')
image_transformer=transforms.Compose(transformer)
img_a = image_transformer(img_a)
img_a = torch.unsqueeze(img_a,dim=0)
real_a = torch.tensor(img_a.type(Tensor))
output = generator(real_a)
output = torch.squeeze(output)
output = output.cpu().detach().numpy()
# plt.imshow(output)
# plt.show()

annotation_image = get_image_mask_from_annotation(img_np.shape[:2], output, 32)
# image = plot_circles_in_image(image,in_tissue_circles,out_tissue_circles,width)
plt.imshow(img_np)
plt.imshow(annotation_image, cmap='binary', alpha=0.5)
plt.show()


# save_image(output.data, save_dir+'test.png', normalize=True)
# output = torch.squeeze(output)
# output = 255* (1.0-output.cpu().detach().numpy())
# output = output.astype(np.uint8)
# # dst = cv2.fastNlMeansDenoising(output,None,10,10,7,21)
# f,a = plt.subplots(1,2)
# a[0].imshow(img_np)
# a[1].imshow(output,cmap='gray')
# plt.show()


# img_a = torch.squeeze(img_a)
# img_a = img_a.numpy()
# index = np.where(output > -0.5)
# x = index[0]
# y = index[1]
# max = img_a.max()
#
# for i,j in zip(x,y):
#     img_a[:,i,j]=[max,max,max]
#
# img_a = np.transpose(img_a, (1, 2, 0))
#
#
# plt.imshow(img_a)
# plt.show()




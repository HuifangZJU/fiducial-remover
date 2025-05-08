import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import time
from matplotlib import pyplot as plt
import numpy as np
import tracemalloc
from dip.models.unet import UNet
from dip.models.skip import skip
import torch
import torch.optim
from dip.utils.inpainting_utils import *

def find_nearest_multiple_of_8(x):
    base = 8
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)
def find_nearest_multiple_of_32(x):
    base = 32
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)

def measure_time_and_memory(func, *args, **kwargs):
    # Reset GPU memory tracker
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Start CPU memory and timer
    tracemalloc.start()
    start_time = time.time()

    # Run function
    result = func(*args, **kwargs)

    # Stop trackers
    end_time = time.time()
    current, peak_cpu = tracemalloc.get_traced_memory()
    peak_gpu = torch.cuda.max_memory_allocated()
    tracemalloc.stop()

    return result, end_time - start_time, peak_cpu / 1024**2, peak_gpu / 1024**2  # Return MB

def save_result(images,filename_prefix,suffix=".png"):

    for i, img_array in enumerate(images):
        # Check if the image is grayscale, RGB, or RGBA
        img_array = img_array.astype(np.uint8)
        if img_array.ndim == 2:  # Grayscale
            img = Image.fromarray(img_array)
        elif img_array.ndim == 3:
            if img_array.shape[2] == 3:  # RGB
                img = Image.fromarray(img_array)
            elif img_array.shape[2] == 4:  # RGBA
                img = Image.fromarray(img_array, 'RGBA')
            else:
                raise ValueError(f"Image at index {i} has an unsupported channel size: {img_array.shape[2]}")
        else:
            raise ValueError(f"Image at index {i} has an unsupported shape: {img_array.shape}")
        # Define the output file name with the prefix
        output_filename = f"{filename_prefix}_{i}" + suffix
        img.save(output_filename)





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

def get_dip_result(input_image, mask_image,num_iter=2000):
    # Desired generation size
    w, h = input_image.size
    new_w = int(find_nearest_multiple_of_32(w))
    new_h = int(find_nearest_multiple_of_32(h))

    img_pil = input_image.resize((new_w, new_h), resample=Image.BICUBIC)
    mask_pil = mask_image.resize((new_w, new_h), resample=Image.NEAREST)


    device = 'cuda'
    img_np = np.asarray(img_pil)
    img_np = img_np/img_np.max()
    mask_np = np.asarray(mask_pil)
    mask_np = (255-mask_np)/255


    # f,a = plt.subplots(1,2)
    # a[0].imshow(img_np)
    # a[1].imshow(mask_np)
    # plt.show()

    print(img_np.shape)


    img_var = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device).float()
    mask_var = torch.from_numpy(mask_np).permute(2, 0, 1).unsqueeze(0).to(device).float()

    input_depth = 3
    net, LR = getInpainter(3, 3)
    net.to('cuda')
    INPUT = 'noise'
    net_input = get_noise(input_depth, INPUT, img_var.shape[-2:])
    # Loss
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    for j in range(1,num_iter+1):
        optimizer.zero_grad()
        net_input = net_input.to(device)
        out = net(net_input)
        loss = mse(out* mask_var, img_var * mask_var)
        loss.backward()
        optimizer.step()
        # print('Iteration %05d ' % j, '\r', end='')
        # show_every=50
        # if j % show_every == 0:
        #     out_np = torch_to_np(out.squeeze().permute(1, 2, 0))
        #     img_np = torch_to_np(img_var.squeeze())
        #     img_np = np.transpose(img_np, (1, 2, 0))
        #     f, a = plt.subplots(2, 2)
        #     a[0, 0].imshow(img_np)
        #     mask_np = torch_to_np(mask_var.squeeze().permute(1, 2, 0))
        #     # mask_np = np.transpose(mask_np, (1, 2, 0))
        #
        #     a[0, 1].imshow(mask_np, cmap='gray')
        #     a[1, 0].imshow(img_np * mask_np)
        #     temp = np.clip(out_np, 0, 1)
        #     a[1, 1].imshow(temp)
        #     plt.show()
    recover_var = net(net_input)
    recover_var = img_var * mask_var + recover_var*(1-mask_var)
    recover_img = torch_to_np(recover_var.squeeze().permute(1, 2, 0))
    recover_img = 255*(recover_img - recover_img.min())/(recover_img.max()-recover_img.min())
    recover_img = recover_img.astype(np.uint8)
    # plt.imshow(recover_img)
    # plt.show()
    #
    # # 4. (Optional) Explicit composite to be 100% sure
    # orig_np = np.array(input_image).astype(np.float32)  # H×W×3
    # out_np = recover_img.astype(np.float32)
    # mask_np = (np.array(mask_image) / 255.0)[..., None]  # H×W×1
    # final_np = out_np * mask_np + orig_np * (1 - mask_np)
    # final = Image.fromarray(final_np.astype(np.uint8))


    return recover_img










def get_stable_diffusion_result(input_image,mask_image,pipe):

    # Desired generation size
    w,h = input_image.size
    new_w = int(find_nearest_multiple_of_8(w))
    new_h = int(find_nearest_multiple_of_8(h))

    init_image = input_image.resize((new_w,new_h), resample=Image.BICUBIC)
    mask_image_resized = mask_image.resize((new_w,new_h), resample=Image.NEAREST)
    result = pipe(
        prompt="Fill the masked region seamlessly.",
        image=init_image,
        mask_image=mask_image_resized,
        width=new_w,
        height=new_h
    )

    # 4. Visualize or save the result
    inpainted_image = result.images[0]

    # multiply the mask to keep the interior textures
    inpainted_image = inpainted_image.resize((w,h), resample=Image.BICUBIC)
    mask_image = mask_image.resize((w,h), Image.NEAREST)
    # 4. (Optional) Explicit composite to be 100% sure
    orig_np = np.array(input_image).astype(np.float32)  # H×W×3
    out_np = np.array(inpainted_image).astype(np.float32)
    mask_np = (np.array(mask_image) / 255.0)[..., None]  # H×W×1
    final_np = out_np * mask_np + orig_np * (1 - mask_np)
    final = Image.fromarray(final_np.astype(np.uint8))
    return final


def get_image_mask_pair(image_path,mask_path,scale):
    # Load original image and mask
    init_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("RGB")
    # mask_image = Image.open(mask_path)

    # Compute new size
    new_size = (int(init_image.width * scale), int(init_image.height * scale))
    # Resize both images
    init_image_resized = init_image.resize(new_size, resample=Image.BICUBIC)
    mask_image_resized = mask_image.resize(new_size, resample=Image.NEAREST)
    return init_image_resized,mask_image_resized




# 1. Load your original image and the corresponding mask
#    The mask should be a black-and-white image where white = region to inpaint, black = keep
image_path = "/media/huifang/data/fiducial/data/1_STDS0000153_stomach_tumor_heterogeneity/7/tissue_hires_image.png"
# mask_path = "/media/huifang/data/fiducial/vispro_masks/103_0.png"
mask_path = "/media/huifang/data/fiducial/data/1_STDS0000153_stomach_tumor_heterogeneity/7/tissue_hires_image_ground_truth.png"


# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     torch_dtype=torch.float16
# )
# pipe = pipe.to("cuda")  # or "cpu" if no GPU


for scale in [0.3,0.5,0.6,0.7,0.83,1,1.2,1.4,1.65]:
# for scale in [0.83]:
# for scale in [0.83]:
    init_image, mask_image = get_image_mask_pair(image_path, mask_path, scale)

    w, h = init_image.size
    new_w = int(find_nearest_multiple_of_32(w))
    new_h = int(find_nearest_multiple_of_32(h))

    img_pil = init_image.resize((new_w, new_h), resample=Image.BICUBIC)
    save_result([np.asarray(img_pil)], "/media/huifang/data/fiducial/temp_result/vispro/restoration/original_" + str(scale))
    continue

    # dip_image, elapsed, peak_cpu, peak_gpu = measure_time_and_memory(get_dip_result,init_image,mask_image)
    # save_result([np.asarray(dip_image)], "/media/huifang/data/fiducial/temp_result/vispro/restoration/new_dip_iter_2000_" + str(scale))
    # print(f"Inpainting image: {elapsed:.4f}s | CPU: {peak_cpu:.2f} MB | GPU: {peak_gpu:.2f} MB")



    # stable_diffusion_image, elapsed, peak_cpu, peak_gpu = measure_time_and_memory(get_stable_diffusion_result, init_image,mask_image,pipe)
    # print(f"Inpainting image: {elapsed:.4f}s | CPU: {peak_cpu:.2f} MB | GPU: {peak_gpu:.2f} MB")
    # # stable_diffusion_image = get_stable_diffusion_result(init_image,mask_image,pipe,scale)
    # save_result([np.asarray(stable_diffusion_image)], "/media/huifang/data/fiducial/temp_result/vispro/restoration/new_stable_diffusion_" + str(scale))
    # test = input()



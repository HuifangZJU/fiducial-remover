import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import time
import numpy as np
# 1. Load your original image and the corresponding mask
#    The mask should be a black-and-white image where white = region to inpaint, black = keep
init_image = Image.open("/media/huifang/data/fiducial/data/1_STDS0000153_stomach_tumor_heterogeneity/7/tissue_hires_image.png").convert("RGB")
mask_image = Image.open("/media/huifang/data/fiducial/data/1_STDS0000153_stomach_tumor_heterogeneity/7/tissue_hires_image_10x.png")



# 2. Load the inpainting pipeline
#    "runwayml/stable-diffusion-inpainting" is a commonly used pretrained model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")  # or "cpu" if no GPU

# 3. Run inpainting with an optional prompt
prompt = "Fill the masked region seamlessly."
# Desired generation size
desired_width = 512
desired_height = 512
start_time = time.time()
result = pipe(
    prompt="A beautiful landscape",
    image=init_image,
    mask_image=mask_image,
    width=desired_width,
    height=desired_height
).images[0]
end_time = time.time()
print(end_time-start_time)
init_image = init_image.resize((512,512),Image.BICUBIC)
mask_image = mask_image.resize((512,512),Image.NEAREST)
# 4. (Optional) Explicit composite to be 100% sure
orig_np = np.array(init_image).astype(np.float32)   # H×W×3
out_np  = np.array(result).astype(np.float32)
mask_np = (np.array(mask_image)/255.0)[...,None]    # H×W×1

final_np = out_np * mask_np + orig_np * (1 - mask_np)
final = Image.fromarray(final_np.astype(np.uint8))
# 4. Visualize or save the result
# inpainted_image.save("/home/huifang/workspace/code/fiducial_remover/revision/stablediffusion/inpainting_result.png")
final.show()

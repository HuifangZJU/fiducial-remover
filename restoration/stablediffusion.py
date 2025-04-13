import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import time

# 1. Load your original image and the corresponding mask
#    The mask should be a black-and-white image where white = region to inpaint, black = keep
init_image = Image.open("/media/huifang/data/fiducial/data/1_STDS0000153_stomach_tumor_heterogeneity/1/tissue_hires_image.png").convert("RGB")
mask_image = Image.open("/media/huifang/data/fiducial/data/1_STDS0000153_stomach_tumor_heterogeneity/1/tissue_hires_image_ground_truth.png").convert("RGB")

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
desired_width = 1792
desired_height = 1792
start_time = time.time()
result = pipe(
    prompt="A beautiful landscape",
    image=init_image,
    mask_image=mask_image,
    width=desired_width,
    height=desired_height
)
end_time = time.time()
print(end_time-start_time)


# 4. Visualize or save the result
inpainted_image = result.images[0]
inpainted_image.save("/home/huifang/workspace/code/fiducial_remover/revision/stablediffusion/inpainting_result.png")
inpainted_image.show()

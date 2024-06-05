import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load a pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Define the transformation to be applied to the input image
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
input_image = Image.open("/home/huifang/workspace/code/fiducial_remover/overlap_annotation/0.png").convert("RGB")
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Move the input to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)

# Perform inference
with torch.no_grad():
    output = model(input_batch)['out'][0]

# The output has shape [num_classes, H, W]
output_predictions = output.argmax(0)

# Visualize the results
plt.imshow(output_predictions.cpu().numpy())
plt.show()

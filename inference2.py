import torch
import models
from timm.models import create_model
from models.modules.mobileone import reparameterize_model
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
# To Train from scratch/fine-tuning
model = create_model("fastvit_t8")
# ... train ...

# Load unfused pre-trained checkpoint for fine-tuning
# or for downstream task training like detection/segmentation
checkpoint = torch.load('model_zoo/fastvit_t8.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

image = Image.open("Capture.png").convert("RGB")

preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])

input_image = preprocess(image).unsqueeze(0)  # Add batch dimension

# 3. Perform inference
with torch.no_grad():
    model.eval()  # Set model to evaluation mode
    output = model(input_image)
    

output_tensor = F.softmax(output, dim=1)




threshold = 0.1

# Filter objects with scores above the threshold
detections = [(idx, score) for idx, score in enumerate(output_tensor[0]) if score > threshold]

# Sort detections by score in descending order
detections.sort(key=lambda x: x[1], reverse=True)

print(output)

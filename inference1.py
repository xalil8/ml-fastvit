import torch
import models
from timm.models import create_model
from models.modules.mobileone import reparameterize_model
from PIL import Image
from torchvision import transforms

import numpy as np
# To Train from scratch/fine-tuning
model = create_model("fastvit_t8")
# ... train ...

# Load unfused pre-trained checkpoint for fine-tuning
# or for downstream task training like detection/segmentation
checkpoint = torch.load('model_zoo/fastvit_t8.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
# ... train ...
# img_np = np.array(Image.open("suv.png").convert('RGB'))
# img_pt = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
# For inference
#model.eval()
#model(img_pt)
#model_inf = reparameterize_model(model)


image = Image.open("suv.png").convert("RGB")

# Define preprocessing transformations (resize, normalize, convert to tensor)
preprocess = transforms.Compose([
    #transforms.Resize((300, 300)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

input_image = preprocess(image).unsqueeze(0)  # Add batch dimension

# 3. Perform inference
with torch.no_grad():
    model.eval()  # Set model to evaluation mode
    output = model(input_image)
    
import torch.nn.functional as F

probabilities = F.softmax(output, dim=1)

print(probabilities)
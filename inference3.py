import torch
import models
from timm.models import create_model
from models.modules.mobileone import reparameterize_model
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# To Train from scratch/fine-tuning
model = create_model("fastvit_t8")
checkpoint = torch.load('model_zoo/fastvit_t8.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

model.eval()      
model_inf = reparameterize_model(model)
model_inf.to(device)



image = Image.open("test_images/humans.jpg").convert("RGB")

# Step 2: Convert to PyTorch tensor and normalize
convert_tensor = transforms.ToTensor()
tensor = convert_tensor(image)

# Step 3: Move tensor to GPU (if available)
tensor = tensor.to(device)

tensor = tensor.unsqueeze(0)
# For inference

torch.cuda.empty_cache()
with torch.no_grad():
    output = model_inf(tensor)
    output_tensor = F.softmax(output, dim=1)


print(output_tensor)
# Use model_inf at test-time
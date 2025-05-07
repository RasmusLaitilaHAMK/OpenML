import torch
import torch.nn as nn
from torchvision import models

# Step 1: Recreate the model architecture (adjust if your model differs)
model = models.resnet18()
num_classes = 8  # Based on your dataset: hyundai, mercedes, etc.
model.fc = nn.Linear(model.fc.in_features, num_classes)

from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and transform the image
img = Image.open("CarLogoDataset/Test/toyota/2r2r.jpg").convert('RGB')
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

# Get class names
class_names = ['hyundai', 'mercedes', 'toyota', 'Volkswagen', 'lexus', 'mazda', 'opel', 'skoda']
print("Predicted class:", class_names[predicted.item()])

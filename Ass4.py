import torch
from torchvision import models, transforms
from PIL import Image

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and class names
checkpoint = torch.load('car_logo_model.pth', map_location=device)

# Rebuild model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['class_names']))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

class_names = checkpoint['class_names']

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
img_path = 'CarLogoDataset/test/mazda/27368.jpg'
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_label = class_names[predicted.item()]

print(f"Predicted class: {predicted_label}")

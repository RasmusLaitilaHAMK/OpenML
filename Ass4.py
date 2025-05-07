# import torch
# from torchvision import models, transforms
# from PIL import Image

# # Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load model and class names
# checkpoint = torch.load('car_logo_model.pth', map_location=device)

# # Rebuild model
# model = models.resnet18()
# model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['class_names']))
# model.load_state_dict(checkpoint['model_state_dict'])
# model = model.to(device)
# model.eval()

# class_names = checkpoint['class_names']

# # Image transform (same as training)
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),  # or RandomResizedCrop(224)
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # Load image
# img_path = 'CarLogoDataset/Test/mazda/27368.jpg'
# image = Image.open(img_path).convert('RGB')
# input_tensor = transform(image).unsqueeze(0).to(device)

# # Predict
# with torch.no_grad():
#     output = model(input_tensor)
#     _, predicted = torch.max(output, 1)
#     predicted_label = class_names[predicted.item()]

# print(f"Predicted class: {predicted_label}")

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

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

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset and loader
test_dataset = datasets.ImageFolder('CarLogoDataset/Test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluation
correct = 0
total = 0

print(f"{'Image':35} | {'True Label':10} | {'Prediction':10}")
print("-" * 65)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for i in range(inputs.size(0)):
            image_path, _ = test_dataset.samples[total + i]
            true_label = test_dataset.classes[labels[i].item()]
            predicted_label = class_names[preds[i].item()]
            print(f"{os.path.basename(image_path):35} | {true_label:10} | {predicted_label:10}")

            if preds[i].item() == labels[i].item():
                correct += 1

        total += labels.size(0)

# Final accuracy
accuracy = correct / total * 100
print("\nAccuracy: {:.2f}% ({}/{})".format(accuracy, correct, total))

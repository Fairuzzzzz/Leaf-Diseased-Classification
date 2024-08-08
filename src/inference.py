import torch
from torchvision import transforms
from PIL import Image
from model import LeafModel
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

num_classes = 15
checkpoint_path = 'path/to/model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = 'dataset/PlantVillage'
class_names = sorted(os.listdir(data_dir))
idx_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}
model = LeafModel.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
model.to(device)
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

image_path = 'path/to/image'
image = preprocess_image(image_path=image_path).to(device)

with torch.no_grad():
    output = model(image)
    predicted_idx = output.argmax(dim=1).item()
    predicted_class = idx_to_class[predicted_idx]

print(f"Predicted Idx: {predicted_idx}")
print(f"Predicted Class: {predicted_class}")

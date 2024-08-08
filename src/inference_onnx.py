import onnxruntime as ort
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image.numpy()

onnx_path = 'leaf_model.onnx'
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

image_path = 'Tomato_Leaf.jpg'
input_data = preprocess_image(image_path=image_path)

outputs = session.run([output_name], {input_name: input_data})
predicted_idx = np.argmax(outputs[0], axis=1)[0]

data_dir = 'dataset/PlantVillage'
class_names = sorted(os.listdir(data_dir))
predicted_class = class_names[predicted_idx]

print(f"Predicted Idx: {predicted_idx}")
print(f"Predicted Class: {predicted_class}")

from io import BytesIO
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image
import os
from scipy.special import softmax
from typing import Any, Dict, Tuple
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ONNXPrediction:
    def __init__(self, onnx_path: str, class_mapping: Dict[int, str]) -> None:
        self.transform = transform
        self.session = ort.InferenceSession(onnx_path)
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        self.input_name = input_info.name
        self.output_name = output_info.name
        self.class_mapping = class_mapping

    def preprocess_image(self, image: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image))
        image = self.transform(image)
        return np.expand_dims(image.numpy(), axis=0)

    def prediction(self, image: bytes) -> Tuple[str, float, int]:
        input_data = self.preprocess_image(image)
        output = self.session.run([self.output_name], {self.input_name: input_data})
        probabilites = softmax(output[0], axis=1)
        predicted_class_idx = int(np.argmax(probabilites))
        predicted_class_prob = float(probabilites[0][predicted_class_idx])
        predicted_class_name = self.class_mapping.get(predicted_class_idx, "Unknown")
        return predicted_class_idx, predicted_class_prob, predicted_class_name

### Leaf Classification using ResNet18

This repository contains a project for leaf classification using pretrained ResNet18 model with PyTorch and PyTorch Ligthning.
The model is fine-tuned to classify different types of leaves from the Plant Village Dataset.

### Table of Contents
- Description
- Requirements
- Installation
- Dataset Preparation
- Training
- Evaluation
- Prediction

### Description
This project uses a pretrained ResNet18 model to classify leaves from the Plant Village Dataset.
The model is fine-tuned to recognize 15 different classes of leaves.

### Requirements
- torch==2.4.0
- torchvision==0.19.0
- pytorch-lightning==2.0.4
- torchmetrics==0.11.4
- tqdm==4.65.0
- Pillow==9.3.0
- onnx==1.16.2
- onnxruntime==1.18.1
- onnxscript==0.1.*

### Installation
1. Clone the repository
```python
git clone https://github.com/Fairuzzzzz/Leaf-Diseased-Classification.git
cd Leaf-Diseased-Classification
```
2. Install Dependencies
```python
pip install requirements.txt
```

### Dataset Preparation
1. Download the PlantVillage Dataset and organize it into subfolders for each class inside dataset folder.
```markdown
dataset/PlantVillage/
    ├── ClassA
    ├── ClassB
    ├── ClassC
```

### Training
To train the model, use the following command:
```python
python src/train.py
```

### Evaluation
To evaluate the model, use the following command:
```python
python src/evaluate.py
```

### Prediction
Change the directory of your images:
```python
image_path = 'path/to/image'
```

To use the model to predict on the validation set, use the following command:
```python
python src/inference.py
```

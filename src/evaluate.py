import torch
from dataset import get_dataloader
from model import LeafModel

data_dir = 'dataset/PlantVillage/'
num_classes = 15
batch_size = 32

_, val_loader = get_dataloader(data_dir=data_dir, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeafModel.load_from_checkpoint('path/to/model', num_classes=num_classes)
model.to(device)
model.eval()

acc = 0.0
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        acc += (outputs.argmax(dim=1) == labels).float().mean().item()
print(f"Validation Accuracy: {acc/len(val_loader):.4f}")

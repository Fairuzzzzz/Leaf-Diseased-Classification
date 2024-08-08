import torch
from model import LeafModel

checkpoint_path = 'path/to/model'
num_classes = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeafModel.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
model.to(device)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224).to(device)

onnx_path = 'leaf_model.onnx'

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model berhasil di konversi ke {onnx_path}")

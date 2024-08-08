import torch
from tqdm import tqdm
import pytorch_lightning as pl
from dataset import get_dataloader
from model import LeafModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

data_dir = 'dataset/PlantVillage/'
num_classes = 15
batch_size = 32

train_loader, val_loader = get_dataloader(data_dir=data_dir, batch_size=batch_size)

model = LeafModel(num_classes=num_classes)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/path',
    filename='leaf-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

trainer = pl.Trainer(max_epochs=20, callbacks = [checkpoint_callback, early_stopping_callback])
trainer.fit(model, train_loader, val_loader)

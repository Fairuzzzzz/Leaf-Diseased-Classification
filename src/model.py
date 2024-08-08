import torch
import pytorch_lightning as pl
import torchvision.models as models
from torchmetrics.functional import accuracy

class LeafModel(pl.LightningModule):
    def __init__(self, num_classes):
        super(LeafModel).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log('Train loss', loss)
        self.log('Train Acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log('Val Loss', loss)
        self.log('Val Acc', acc)

        return loss

    def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

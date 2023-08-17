
from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from utils.get_device import find_lr

class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, bias=False, stride=1, padding=1, pool=False, dropout=0.):
        super(ConvLayer, self).__init__()

        layers = list()
        layers.append(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, bias=bias, stride=stride, padding=padding,
                      padding_mode='replicate')
        )
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.all_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.all_layers(x)


class CustomLayer(nn.Module):
    def __init__(self, input_channels, output_channels, pool=True, res=2, dropout=0.):
        super(CustomLayer, self).__init__()

        self.pool_layer = ConvLayer(input_channels, output_channels, pool=pool, dropout=dropout)
        self.res_block = None
        if res > 0:
            layers = list()
            for i in range(0, res):
                layers.append(ConvLayer(output_channels, output_channels, pool=False, dropout=dropout))
            self.res_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool_layer(x)
        if self.res_block is not None:
            x_ = x
            x = self.res_block(x)
            x = x + x_
        return x


class CustomResNet(LightningModule):
    def __init__(self, dataset, dropout=0.05, max_epochs=24):
        super(CustomResNet, self).__init__()

        self.dataset = dataset

        self.network = nn.Sequential(
            CustomLayer(3, 64, pool=False, res=0, dropout=dropout),
            CustomLayer(64, 128, pool=True, res=2, dropout=dropout),
            CustomLayer(128, 256, pool=True, res=0, dropout=dropout),
            CustomLayer(256, 512, pool=True, res=2, dropout=dropout),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=10)

        self.max_epochs = max_epochs

    def forward(self, x):
        return self.network(x)

    def common_step(self, batch, mode):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        acc_metric = getattr(self, f'{mode}_accuracy')
        acc_metric(logits, y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, 'train')
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, 'val')
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, list):
            x, _ = batch
        else:
            x = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-7, weight_decay=1e-2)
        best_lr = find_lr(self, self.train_dataloader(), optimizer, self.criterion)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=best_lr,
            steps_per_epoch=len(self.dataset.train_loader),
            epochs=self.max_epochs,
            pct_start=5/self.max_epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def prepare_data(self):
        self.dataset.download()

    def train_dataloader(self):
        return self.dataset.train_loader

    def val_dataloader(self):
        return self.dataset.test_loader

    def predict_dataloader(self):
        return self.val_dataloader()
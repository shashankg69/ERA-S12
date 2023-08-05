from typing import Any
import torch
from torch import Tensor, optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, Metric
from torch_lr_finder import LRFinder


class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, bias= False, stride = 1, padding = 1, pool = False, dropout =0):
        super(ConvLayer, self).__init__()

        layers = list()
        layers.append(
            nn.Conv2d(input_channels, output_channels, 3, bias= bias, stride = stride, padding= padding, padding_mode='replicate')
        )
        if pool:
            layers.append(
                nn.MaxPool2d(2,2)
            )
        layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.layer_outputs = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer_outputs(x)
    

class CustomLayer(nn.Module):
    def __init__(self, input_channels, output_channels, pool = True, res= 2, dropout = 0):
        super(CustomLayer, self).__init__()

        self.conv_layer = ConvLayer(input_channels, output_channels, pool=pool, dropout=dropout)
        self.res_layer = None
        if res > 0:
            layers = list()
            for i in range(0, res):
                layers.append(
                    ConvLayer(output_channels, output_channels, pool=False, dropout=dropout)
                    )
            self.res_layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.res_layer is not None:
            x_ = x
            x = self.res_layer(x)
            x = x + x_
        return x
    
class CustomResNet(LightningModule):
    def __init__(self, dataset, dropout=0.05, max_epochs=24):
        super(CustomResNet, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.train_accuracy = Acc_Metric()
        self.val_accuracy = Acc_Metric()
        self.layer_outputs = nn.Sequential(
            CustomLayer(input_channels= 3, output_channels= 64, pool= False, res = 0, dropout=dropout),
            CustomLayer(input_channels= 64, output_channels= 128, pool= True, res = 2, dropout=dropout),
            CustomLayer(input_channels= 128, output_channels= 256, pool= True, res = 0, dropout=dropout),
            CustomLayer(input_channels= 256, output_channels= 512, pool= True, res = 2, dropout=dropout),
            nn.MaxPool2d(4,4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.max_epochs = max_epochs
        self.epoch_counter = 1

    def forward(self, x):
        x = self.layer_outputs(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        batch_len = y.numel()
        self.train_loss.update(loss, batch_len)
        self.train_accuracy.update(y_pred, y)
        return loss
    
    def on_train_epoch_end(self):
        print(f"Epoch: {self.epoch_counter}, ||Train|| : Training_Loss: {self.train_loss.compute():0.4f}, Training_Accuracy: "
          f"{self.train_accuracy.compute():0.2f}")
        self.train_loss.reset()
        self.train_accuracy.reset()
        self.epoch_counter += 1
    
    def vaidation_step(self,batch, batch_idx):
        x, y = batch
        batch_len = y.numel()
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        self.val_loss.update(loss, batch_len)
        self.val_accuracy.update(y_pred, y)
        self.log("val_step_loss", loss, prog_bar=True, logger=True)
        self.log("val_step_acc", self.val_accuracy.compute(), prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_end(self):
        print(f"Epoch: {self.epoch_counter}, ||Validation|| : Validation_Loss: {self.val_loss.compute():0.4f}, Validation_Accuracy: "
              f"{self.val_accuracy.compute():0.2f}")
        self.val_loss.reset()
        self.val_accuracy.reset()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= 1e-7, weight_decay= 1e-2)
        best_lr = self.find_lr(optimizer)
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch  
        predictions = self.forward(x) 
        return predictions
    
    def find_lr(self,optimizer):
        lr_finder = LRFinder(self, optimizer, self.criterion)
        lr_finder.range_test(self.dataset.train_loader, end_lr=0.1, num_iter=100, step_mode='exp')
        _, best_lr = lr_finder.plot()
        lr_finder.reset
        return best_lr
    

class Acc_Metric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        preds = preds.argmax(dim=1)
        total = target.numel()
        self.correct += preds.eq(target).sum()
        self.total += total

    def compute(self):
        return 100 * self.correct.float() / self.total         
                


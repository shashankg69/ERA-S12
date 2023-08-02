import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


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

        self.all_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.all_layers(x)
    

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

        
            
                


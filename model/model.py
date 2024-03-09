import torch
from torch import nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, in_channels: int, squeeze1x1: int, expand1x1: int, expand3x3: int) -> None:
        super().__init__()
        self.s1x1 = nn.Conv2d(in_channels, squeeze1x1, 1)
        self.e1x1 = nn.Conv2d(squeeze1x1, expand1x1, 1)
        self.e3x3 = nn.Conv2d(squeeze1x1, expand3x3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.s1x1(x))
        x = torch.cat([F.relu(self.e1x1(x)), F.relu(self.e3x3(x))], dim=1)
        return x
    
class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, 2) 
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.fire2 = FireModule(96, 16, 64, 64) 
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, 2)
        self.fire5 = FireModule(256, 32, 128, 128)
        self.fire6 = FireModule(256, 48, 192, 192)
        self.fire7 = FireModule(384, 48, 192, 192)
        self.fire8 = FireModule(384, 64, 256, 256)
        self.maxpool8 = nn.MaxPool2d(3, 2)
        self.fire9 = FireModule(512, 64, 256, 256)
        self.conv10 = nn.Conv2d(512, num_classes, 1)
        self.avgpool10 = nn.AvgPool2d(13, 1, 1)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.conv10:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        x = self.fire9(x)
        x = F.dropout((x), 0.5)
        x = F.relu(self.conv10(x))
        x = self.avgpool10(x)
        return x
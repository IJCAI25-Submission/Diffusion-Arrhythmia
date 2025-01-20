import torch.nn as nn
from efficientnet_pytorch import EfficientNet as efficientnet
from torchvision import models
import torch

def EfficientNetB4():
    model = efficientnet.from_name("efficientnet-b4")

    first_conv_layer = model._conv_stem
    original_weight = first_conv_layer.weight

    new_first_conv_layer = nn.Conv2d(1, first_conv_layer.out_channels,
                                     kernel_size=first_conv_layer.kernel_size,
                                     stride=first_conv_layer.stride,
                                     padding=first_conv_layer.padding,
                                     bias=first_conv_layer.bias)

    new_weight = original_weight.mean(dim=1, keepdim=True)

    new_first_conv_layer.weight = nn.Parameter(new_weight)

    model._conv_stem = new_first_conv_layer

    return model

class downStreamClassifier(nn.Module):
    def __init__(self, baseModel, num_classes):
        super(downStreamClassifier, self).__init__()
        self.baseModel = baseModel
        self.baseModel._fc = nn.Linear(self.baseModel._fc.in_features, num_classes)

    def forward(self, x):
        x = self.baseModel(x)
        return x
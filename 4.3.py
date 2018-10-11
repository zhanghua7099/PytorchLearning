import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        layer1 = nn.Sequential()
        self.layer1 = layer1

        layer2 = nn.Sequential()
        self.layer2 = layer2

        layer3 = nn.Sequential()
        self.layer3 = layer3

        layer4 = nn.Sequential()
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out
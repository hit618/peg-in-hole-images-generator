# https://github.com/usuyama/pytorch-unet
import torch
from torch import nn
import torchvision
from ConvLSTMModel import ConvLSTM

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def linearRelu(in_features,out_features,bias):
    return nn.Sequential(
        nn.Linear(in_features,out_features,bias=bias),
        nn.ReLU(inplace=True),
    )

def linearTanh(in_features,out_features,bias):
    return nn.Sequential(
        nn.Linear(in_features,out_features,bias=bias),
        nn.Tanh(),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class, feat_preultimate=64, inputLayersNum=3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        if self.inputLayersNum == 4:
            self.preconv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        # self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc0 = linearRelu(4608,4096,True)
        # self.fc1 = linearRelu(4096, 512, True)
        # self.fc2 = linearRelu(512, 1, True)
        # self.conv_original_size0 = convrelu(3, 64, 3, 1)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, feat_preultimate, 3, 1)

        self.conv_last = nn.Conv2d(feat_preultimate, n_class, 1)
        for p in self.base_model.conv1.parameters():
            p.requires_grad = False

    def forward(self, input):
        # x_original = self.conv_original_size0(input)
        # x_original = self.conv_original_size1(x_original)
        if self.inputLayersNum == 4:
            input = self.preconv(input)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)

        # y = self.maxpool0(layer4)
        # y = nn.Flatten(y)
        # y = self.fc0(y)
        # y = self.fc1(y)
        # y = self.fc2(y)

        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResNetUNet2(nn.Module):
    def __init__(self, n_class, feat_preultimate=64, inputLayersNum=3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        if self.inputLayersNum == 4:
            self.preconv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc0 = linearRelu(4608, 512, True)
        # self.fc1 = linearRelu(4096, 512, True)
        self.fc2 = linearRelu(512, 2, True)
        # self.conv_original_size0 = convrelu(3, 64, 3, 1)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, feat_preultimate, 3, 1)

        self.conv_last = nn.Conv2d(feat_preultimate, n_class, 1)
        for p in self.base_model.conv1.parameters():
            p.requires_grad = False

    def forward(self, input):
        # x_original = self.conv_original_size0(input)
        # x_original = self.conv_original_size1(x_original)
        if self.inputLayersNum == 4:
            input = self.preconv(input)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)

        y = self.maxpool0(layer4)
        y = self.flatten(y)
        y = self.fc0(y)
        # y = self.fc1(y)
        y = self.fc2(y)
        # y = (y + 1) / 2

        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out, y

class ConvLSTMUNet(nn.Module):
    def __init__(self, n_class, feat_preultimate=64, inputLayersNum=3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        if self.inputLayersNum == 4:
            self.preconv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.convLSTM = ConvLSTM(3, 16, (3,3), 2, True, True, False)

        self.base_layers[0] = nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),bias=False)

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, feat_preultimate, 3, 1)

        self.conv_last = nn.Conv2d(feat_preultimate, n_class, 1)
        for p in self.base_model.conv1.parameters():
            p.requires_grad = False

    def forward(self, input):
        # x_original = self.conv_original_size0(input)
        # x_original = self.conv_original_size1(x_original)
        if self.inputLayersNum == 4:
            input = self.preconv(input)
        layer_output, last_states = self.convLSTM(input)
        input = last_states[0][0]
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

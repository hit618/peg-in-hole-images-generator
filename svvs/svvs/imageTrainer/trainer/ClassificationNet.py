import torch
from torch import nn
# import torch.nn.functional as F
import torchvision
from model import Model

import numpy as np
import albumentations as A
_augs = [
    # A.ColorJitter(brightness=0.4, contrast=0.4),
    A.ISONoise(),
    A.GaussNoise(),
    A.GaussianBlur(blur_limit=(1, 9)),
]

class VggNet(nn.Module):
    def __init__(self,inputLayersNum=4, classNum = 3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.base_model = torchvision.models.vgg16(pretrained=True)
        # print(self.base_model)
        if self.inputLayersNum==4:
            self.preConv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.classifier[6] = nn.Linear(4096, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        if self.inputLayersNum == 4:
            input = self.preConv(input)
        out = self.base_model(input)

        return out

class SegmentVggNet(nn.Module):
    def __init__(self,inputLayersNum=3, classNum = 3, checkpoint_path=None, augs=False):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        self.augs = augs
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.segmentModel = Model.load_from_checkpoint(checkpoint_path, loss='mse')
        self.base_model = torchvision.models.vgg16(pretrained=True)
        # print(self.base_model)
        self.preConv = nn.Conv2d(2, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.classifier[6] = nn.Linear(4096, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        with torch.no_grad():
            input = self.segmentModel(input)
        if self.augs:
            input = self.imageAugs(input)
        input = self.preConv(input)
        out = self.base_model(input)
        return out

    def getSegment(self,input):
        mask = self.segmentModel(input)
        if self.augs:
            mask = self.imageAugs(mask)
        return mask

    def imageAugs(self,input):
        a = _augs.copy()
        # np.random.shuffle(a)
        a = A.Compose(a)
        image = input.cpu().detach().numpy()*255
        np.clip(image, 0, 255, out=image)
        image = image.astype(np.uint8)
        inageShape = image.shape
        inageShape = np.array(inageShape)
        image_tensor = torch.zeros(image.shape)
        inageShape[1] = 3
        zeroImage = np.zeros(tuple(inageShape))
        zeroImage[:,0:2,:,:]=image
        zeroImage = zeroImage.astype(np.uint8)
        img = []
        for i in range(len(zeroImage)):
            image_i=zeroImage[i].transpose((1,2,0))
            image_i=a(image=image_i)['image']
            image_i = image_i[:,:,0:2]/255
            # image_i=np.array(image_i).transpose((2,0,1))
            # img.append(image_i[0:2])
            image_tensor_i = torchvision.transforms.functional.to_tensor(image_i)
            # image_tensor_i = image_tensor_i.unsqueeze(0)
            image_tensor[i] = image_tensor_i
        # img = np.array(img)/255
        # out = torchvision.transforms.functional.to_tensor(img)
        # out = torch.from_numpy(img.astype(float)).to('cuda:0')
        image_tensor = image_tensor.to('cuda:0')
        return image_tensor

class MobileNet(nn.Module):
    def __init__(self,inputLayersNum=4, classNum = 3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.base_model = torchvision.models.mobilenet_v3_large(pretrained=True)
        # print(self.base_model)
        if self.inputLayersNum==4:
            self.preConv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.classifier[3] = nn.Linear(1280, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        if self.inputLayersNum == 4:
            input = self.preConv(input)
        out = self.base_model(input)

        return out

class SegmentMobileNet(nn.Module):
    def __init__(self,inputLayersNum=3, classNum = 3, checkpoint_path=None, augs=True):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        self.augs = augs
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.segmentModel = Model.load_from_checkpoint(checkpoint_path, loss='mse')
        self.base_model = torchvision.models.mobilenet_v3_large(pretrained=True)
        # print(self.base_model)
        self.preConv = nn.Conv2d(2, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.classifier[3] = nn.Linear(1280, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        with torch.no_grad():
            input = self.segmentModel(input)
        if self.augs:
            # print(self.augs)
            input = self.imageAugs(input)
        input = self.preConv(input)
        out = self.base_model(input)
        return out

    def getSegment(self,input):
        mask = self.segmentModel(input)
        if self.augs:
            mask = self.imageAugs(mask)
        return mask

    def imageAugs(self,input):
        a = _augs.copy()
        # np.random.shuffle(a)
        a = A.Compose(a)
        image = input.cpu().detach().numpy()*255
        np.clip(image, 0, 255, out=image)
        image = image.astype(np.uint8)
        inageShape = image.shape
        inageShape = np.array(inageShape)
        image_tensor = torch.zeros(image.shape)
        inageShape[1] = 3
        zeroImage = np.zeros(tuple(inageShape))
        zeroImage[:,0:2,:,:]=image
        zeroImage = zeroImage.astype(np.uint8)
        img = []
        for i in range(len(zeroImage)):
            image_i=zeroImage[i].transpose((1,2,0))
            image_i=a(image=image_i)['image']
            image_i = image_i[:,:,0:2]/255
            # image_i=np.array(image_i).transpose((2,0,1))
            # img.append(image_i[0:2])
            image_tensor_i = torchvision.transforms.functional.to_tensor(image_i)
            # image_tensor_i = image_tensor_i.unsqueeze(0)
            image_tensor[i] = image_tensor_i
        # img = np.array(img)/255
        # out = torchvision.transforms.functional.to_tensor(img)
        # out = torch.from_numpy(img.astype(float)).to('cuda:0')
        image_tensor = image_tensor.to('cuda:0')
        return image_tensor

class SegmentMobileNetV2(nn.Module):
    def __init__(self,inputLayersNum=3, classNum = 3, checkpoint_path=None, augs=False):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        self.augs = augs
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.segmentModel = Model.load_from_checkpoint(checkpoint_path, loss='mse', inputLayersNum=self.inputLayersNum)
        self.base_model = torchvision.models.mobilenet_v3_large(pretrained=True)
        # print(self.base_model)
        self.preConv = nn.Conv2d(2+self.inputLayersNum, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.classifier[3] = nn.Linear(1280, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        with torch.no_grad():
            mask = self.segmentModel(input)
        if self.augs:
            mask = self.imageAugs(mask)
        mask = torch.cat([mask, input], 1)
        out = self.preConv(mask)
        out = self.base_model(out)
        return out

    def getSegment(self, input):
        mask = self.segmentModel(input)
        if self.augs:
            mask = self.imageAugs(mask)
        return mask

    def imageAugs(self,input):
        a = _augs.copy()
        # np.random.shuffle(a)
        a = A.Compose(a)
        image = input.cpu().detach().numpy()*255
        np.clip(image, 0, 255, out=image)
        image = image.astype(np.uint8)
        inageShape = image.shape
        inageShape = np.array(inageShape)
        image_tensor = torch.zeros(image.shape)
        inageShape[1] = 3
        zeroImage = np.zeros(tuple(inageShape))
        zeroImage[:,0:2,:,:]=image
        zeroImage = zeroImage.astype(np.uint8)
        img = []
        for i in range(len(zeroImage)):
            image_i=zeroImage[i].transpose((1,2,0))
            image_i=a(image=image_i)['image']
            image_i = image_i[:,:,0:2]/255
            # image_i=np.array(image_i).transpose((2,0,1))
            # img.append(image_i[0:2])
            image_tensor_i = torchvision.transforms.functional.to_tensor(image_i)
            # image_tensor_i = image_tensor_i.unsqueeze(0)
            image_tensor[i] = image_tensor_i
        # img = np.array(img)/255
        # out = torchvision.transforms.functional.to_tensor(img)
        # out = torch.from_numpy(img.astype(float)).to('cuda:0')
        image_tensor = image_tensor.to('cuda:0')
        return image_tensor

class ResNet18(nn.Module):
    def __init__(self,inputLayersNum=4, classNum = 3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.base_model = torchvision.models.resnet18(pretrained=True)
        # print(self.base_model)
        if self.inputLayersNum==4:
            self.preConv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.fc.add_module('add_linear', nn.Linear(1000, 256))
        # self.base_model.fc.add_module('add_linear', nn.Linear(256, 3))
        self.base_model.fc = nn.Linear(512, self.classNum)
        # self.afterLayer = nn.Linear(64, 3)

    def forward(self, input):
        if self.inputLayersNum == 4:
            input = self.preConv(input)
        out = self.base_model(input)
        # out = nn.functional.relu(out)
        # out = self.afterLayer(out)

        return out

class SegmentResNet18(nn.Module):
    def __init__(self,inputLayersNum=3, classNum = 3, checkpoint_path=None):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        self.segmentModel = Model.load_from_checkpoint(checkpoint_path, loss = 'mse')
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.base_model = torchvision.models.resnet18(pretrained=True)
        # print(self.base_model)
        self.preConv = nn.Conv2d(2, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # if self.inputLayersNum==4:
        #     self.preConv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.fc.add_module('add_linear', nn.Linear(1000, 256))
        # self.base_model.fc.add_module('add_linear', nn.Linear(256, 3))
        self.base_model.fc = nn.Linear(512, self.classNum)
        # self.afterLayer = nn.Linear(64, 3)

    def forward(self, input):
        with torch.no_grad():
            input = self.segmentModel(input)
        input = self.preConv(input)
        out = self.base_model(input)
        return out

class ResNet50(nn.Module):
    def __init__(self,inputLayersNum=4, classNum = 3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.base_model = torchvision.models.resnet50(pretrained=True)
        # print(self.base_model)
        if self.inputLayersNum==4:
            self.preConv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.fc = nn.Linear(2048, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        if self.inputLayersNum == 4:
            input = self.preConv(input)
        out = self.base_model(input)

        return out

class regNet(nn.Module):
    def __init__(self,inputLayersNum=4, classNum = 3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.base_model = torchvision.models.regnet_y_800mf(pretrained=True)
        # print(self.base_model)
        if self.inputLayersNum==4:
            self.preConv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.fc = nn.Linear(784, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        if self.inputLayersNum == 4:
            input = self.preConv(input)
        out = self.base_model(input)

        return out

class efficientNet(nn.Module):
    def __init__(self,inputLayersNum=4, classNum = 3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.base_model = torchvision.models.efficientnet_v2_s(pretrained=True)
        # print(self.base_model)
        if self.inputLayersNum==4:
            self.preConv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.classifier[1] = nn.Linear(1280, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        if self.inputLayersNum == 4:
            input = self.preConv(input)
        out = self.base_model(input)

        return out

class denseNet121(nn.Module):
    def __init__(self,inputLayersNum=4, classNum = 3):
        super().__init__()
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        # self.base_model = torchvision.models.vgg11(pretrained=True)
        self.base_model = torchvision.models.densenet121(pretrained=True)
        # print(self.base_model)
        if self.inputLayersNum==4:
            self.preConv = nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.base_model.classifier.add_module('add_linear', nn.Linear(1000, 3))
        self.base_model.classifier = nn.Linear(1024, self.classNum)
        # for p in self.base_model.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        if self.inputLayersNum == 4:
            input = self.preConv(input)
        out = self.base_model(input)

        return out
import enum

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
from DualArmAsm.imageTrainer.trainer.ClassificationNet import VggNet,SegmentVggNet,SegmentMobileNetV2,MobileNet,SegmentMobileNet,ResNet18,ResNet50,regNet,efficientNet,denseNet121,SegmentResNet18

class ClassificationModel(pl.LightningModule):
    def __init__(self, loss='CrossEntropy',inputLayersNum=4, classNum=3, networkName='VggNet', epochs = 30,segment_checkpoint_path=None, augs = False):
        super().__init__()
        self.loss = loss
        self.inputLayersNum = inputLayersNum
        self.classNum = classNum
        self.networkName=networkName
        self.save_hyperparameters()
        self.augs = augs
        if self.networkName == 'VggNet':
            self.Net = VggNet(inputLayersNum=self.inputLayersNum, classNum=self.classNum)
        elif self.networkName == 'SegmentVggNet':
            self.Net = SegmentVggNet(inputLayersNum=self.inputLayersNum, classNum=self.classNum,
                                     checkpoint_path=segment_checkpoint_path, augs=self.augs)
        elif self.networkName == 'MobileNet':
            self.Net = MobileNet(inputLayersNum=self.inputLayersNum, classNum=self.classNum)
        elif self.networkName == 'SegmentMobileNet':
            self.Net = SegmentMobileNet(inputLayersNum=self.inputLayersNum, classNum=self.classNum,
                                     checkpoint_path=segment_checkpoint_path, augs=self.augs)
        elif self.networkName == 'SegmentMobileNetV2':
            self.Net = SegmentMobileNetV2(inputLayersNum=self.inputLayersNum, classNum=self.classNum,
                                     checkpoint_path=segment_checkpoint_path, augs=self.augs)
        elif self.networkName == 'ResNet18':
            self.Net = ResNet18(inputLayersNum=self.inputLayersNum, classNum=self.classNum)
        elif self.networkName == 'SegmentResNet18':
            self.Net = SegmentResNet18(inputLayersNum=self.inputLayersNum, classNum=self.classNum,
                                       checkpoint_path=segment_checkpoint_path)
        elif self.networkName == 'ResNet50':
            self.Net = ResNet50(inputLayersNum=self.inputLayersNum, classNum=self.classNum)
        elif self.networkName == 'regNet':
            self.Net = regNet(inputLayersNum=self.inputLayersNum, classNum=self.classNum)
        elif self.networkName == 'efficientNet':
            self.Net = efficientNet(inputLayersNum=self.inputLayersNum, classNum=self.classNum)
        elif self.networkName == 'denseNet121':
            self.Net = denseNet121(inputLayersNum=self.inputLayersNum, classNum=self.classNum)
        else:
            raise ValueError
        self.save_hyperparameters('loss')

    def forward(self, x):
        x = self.Net(x)  # (B, 2, H, W)
        return x

    def getSegment(self,x):
        x = self.Net.getSegment(x)  # (B, 2, H, W)
        return x

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        def lr(i):
            # warmup
            lr_ = min(i / 100, 1.)
            return lr_

        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lr),
                interval='step',
            ),
        )

    def step(self, batch, log_name, log_pbar=False):
        x, y = batch
        lgts = self.Net(x)
        # self.image_show(batch.copy())
        if self.loss == 'CrossEntropy':
            loss = F.cross_entropy(lgts, y)
            # self.image_show(batch.copy(),lgts.detach())
        else:
            raise ValueError()
        self.log(f'loss_{log_name}', loss, prog_bar=log_pbar)
        # self.log_dict({'inputLayersNum':self.inputLayersNum,'classNum':self.classNum})
        return loss

    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        return self.step(batch, 'val', True)

    def image_show(self, batch, lgts):
        x, y = batch
        x= x.permute(0, 2, 3, 1)
        img = torch.squeeze(x[0]).cpu().detach().numpy()
        # matplotlib.image.imsave('image_record/img.png', img)
        plt.figure()
        plt.imshow(img)

        lable = torch.squeeze(y[0]).cpu().detach().numpy()
        # matplotlib.image.imsave('image_record/lable.png', lable)
        plt.figure()
        plt.imshow(lable[0])
        plt.figure()
        plt.imshow(lable[1])

        lgts = torch.squeeze(lgts[0]).cpu().numpy()
        # matplotlib.image.imsave('image_record/lgts.png', lgts)
        plt.figure()
        plt.imshow(lgts[0])
        plt.figure()
        plt.imshow(lgts[1])

        plt.show()
import enum

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
from DualArmAsm.imageTrainer.trainer.unet import ResNetUNet, ResNetUNet2, ConvLSTMUNet

class Model(pl.LightningModule):
    def __init__(self, loss='mse',n_class=2, inputLayersNum=3):
        super().__init__()
        assert loss in ['mse', 'bce']
        self.loss = loss
        self.unet = ResNetUNet(n_class=n_class, inputLayersNum=inputLayersNum)
        # self.save_hyperparameters('loss')
        self.save_hyperparameters()

    def forward(self, x):
        x = self.unet(x)  # (B, 2, H, W)
        if self.loss == 'bce':
            x = torch.sigmoid(x)
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
        lgts = self.unet(x)
        # self.image_show(batch.copy())
        if self.loss == 'mse':
            loss = F.mse_loss(lgts, y)
            # self.image_show(batch.copy(),lgts.detach())
        elif self.loss == 'bce':
            loss = F.binary_cross_entropy_with_logits(lgts, y)
        else:
            raise ValueError()
        self.log(f'loss_{log_name}', loss, prog_bar=log_pbar)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        return self.step(batch, 'val', True)

    def image_show(self,batch,lgts):
        x, y = batch
        x= x.permute(0, 2, 3, 1)
        img = torch.squeeze(x[0]).cpu().detach().numpy()
        # matplotlib.image.imsave('image_record/img.png', img)
        plt.figure()
        plt.imshow(img[:,:,0:3])

        lable = torch.squeeze(y[0]).cpu().detach().numpy()
        # matplotlib.image.imsave('image_record/lable.png', lable)
        plt.figure()
        plt.imshow(lable[0])
        # plt.figure()
        # plt.imshow(lable[1])

        # lgts = torch.squeeze(lgts[0]).cpu().numpy()
        # matplotlib.image.imsave('image_record/lgts.png', lgts)
        # plt.figure()
        # plt.imshow(lgts)
        # plt.figure()
        # plt.imshow(lgts[1])

        plt.show()

class Model2(Model):
    def __init__(self, loss='mse',n_class=2, inputLayersNum=3):
        super().__init__(loss=loss,n_class=n_class, inputLayersNum=inputLayersNum)
        self.unet = ResNetUNet2(n_class=n_class, inputLayersNum=inputLayersNum)

    def forward(self, x):
        x, y = self.unet(x)  # (B, 2, H, W)
        if self.loss == 'bce':
            x = torch.sigmoid(x)
        return x, y

    def step(self, batch, log_name, log_pbar=False):
        x, y = batch
        lgts, markers = self.unet(x)
        # self.image_show(batch.copy())
        markerLableOriginal = y[:,0,:,:]
        markerLableSum = torch.sum(markerLableOriginal,dim=(1,2), keepdim=False)
        # markerLable[markerLableSum > 0.2] = [0,1]
        # markerLable[markerLableSum < 0.2] = [1,0]
        bitchSize = len(lgts)
        markerLable = torch.zeros((bitchSize,2)).to('cuda:0')
        for i in range(bitchSize):
            if markerLableSum[i] > 0.2:
                markerLable[i,:] = torch.tensor([0, 1])
            else:
                markerLable[i,:] = torch.tensor([1, 0])
        if self.loss == 'mse':
            loss1 = F.mse_loss(lgts, y)
            # loss2 = F.mse_loss(markers, markerLable)
            loss2 = F.cross_entropy(markers, markerLable)
            loss = loss1 + 0.005 * loss2
            # self.image_show(batch.copy(),lgts.detach())
        elif self.loss == 'bce':
            loss = F.binary_cross_entropy_with_logits(lgts, y)
        else:
            raise ValueError()
        self.log(f'loss_{log_name}', loss, prog_bar=log_pbar)
        return loss




class LstmModel(pl.LightningModule):
    def __init__(self,loss='mse',n_class=2, inputLayersNum=3, sequenceLength = 5):
        # super().__init__(loss=loss,n_class=n_class, inputLayersNum=inputLayersNum)
        super().__init__()
        assert loss in ['mse', 'bce']
        self.loss = loss
        self.save_hyperparameters()
        self.convLSTMUNet = ConvLSTMUNet(n_class=n_class, inputLayersNum=inputLayersNum)
        self.sequenceLength = sequenceLength
        self.inputLayersNum = inputLayersNum

    def forward(self, x):
        x = x.view(-1,self.sequenceLength,self.inputLayersNum,224,224)
        x = self.convLSTMUNet(x)  # (B, S, 2, H, W)
        if self.loss == 'bce':
            x = torch.sigmoid(x)
        return x

    def step(self, batch, log_name, log_pbar=False):
        x, y = batch
        x = x.view(-1, self.sequenceLength, self.inputLayersNum, 224, 224)
        lgts = self.convLSTMUNet(x)
        y = y.view(-1,self.sequenceLength,2,224,224)
        y = y[:,-1,:,:,:]
        # self.image_show(batch.copy())
        if self.loss == 'mse':
            loss = F.mse_loss(lgts, y)
            # self.image_show(batch.copy(), x.detach(), lgts.detach(), y.detach())
        elif self.loss == 'bce':
            loss = F.binary_cross_entropy_with_logits(lgts, y)
        else:
            raise ValueError()
        self.log(f'loss_{log_name}', loss, prog_bar=log_pbar)
        return loss
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
    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        return self.step(batch, 'val', True)

    def image_show(self,batch, x, lgts, y):
        # x, y = batch
        x = x[0].permute(0, 2, 3, 1)
        img = torch.squeeze(x[4]).cpu().detach().numpy()

        plt.figure()
        plt.imshow(img)

        lable = torch.squeeze(y[0]).cpu().detach().numpy()
        # matplotlib.image.imsave('image_record/lable.png', lable)
        plt.figure()
        plt.imshow(lable[0])
        # plt.figure()
        # plt.imshow(lable[1])

        # lgts = torch.squeeze(lgts).cpu().numpy()
        # matplotlib.image.imsave('image_record/lgts.png', lgts)
        # plt.figure()
        # plt.imshow(lgts)
        # plt.figure()
        # plt.imshow(lgts[1])

        plt.show()




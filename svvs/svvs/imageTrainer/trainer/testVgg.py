import argparse
import multiprocessing
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from ClassificationModel import ClassificationModel
from pathlib import Path
from DualArmAsm.imageTrainer.makeDataset.dataset import SynthDataset, VggDataset

parser = argparse.ArgumentParser()
parser.add_argument("--image-num", type=list, default=[100,100,100])
# parser.add_argument("--image-num", type=list, default=[30,30,30])
parser.add_argument('--loss', type=str, default='CrossEntropy')
parser.add_argument('--inputLayersNum', type=int, default=3)
parser.add_argument('--augs', type=bool, default=False)
parser.add_argument('--networkName', type=str, default='VggNet')# SegmentVggNet SegmentMobileNet ResNet50 ResNet18 SegmentResNet18 MobileNet VggNet regNet efficientNet denseNet121
# parser.add_argument('--checkpoint-path', type=str, default='lightning_logs/version_31/checkpoints/epoch=2-step=303.ckpt')
# parser.add_argument('--checkpoint-path', type=str, default='lightning_logs/version_32/checkpoints/epoch=18-step=1995.ckpt')
# parser.add_argument('--checkpoint-path', type=str, default='lightning_logs/version_41/checkpoints/epoch=22-step=2576.ckpt')
# parser.add_argument('--checkpoint-path', type=str, default='lightning_logs/version_42/checkpoints/epoch=7-step=1496.ckpt')
# parser.add_argument('--checkpoint-path', type=str, default='lightning_logs/version_49/checkpoints/epoch=49-step=5600.ckpt')
# parser.add_argument('--checkpoint-path', type=str, default='lightning_logs/version_50/checkpoints/epoch=29-step=3360.ckpt')
# parser.add_argument('--checkpoint-path1', type=str, default='lightning_logs/version_52/checkpoints/epoch=29-step=5610.ckpt')# 4层输入
parser.add_argument('--checkpoint-path2', type=str, default='lightning_logs/version_218/checkpoints/epoch=29-step=5610.ckpt')# 3层输入
# parser.add_argument('--checkpoint-path1', type=str, default='lightning_logs/version_64/checkpoints/epoch=29-step=5610.ckpt')# 4层输入
parser.add_argument('--checkpoint-path1', type=str, default='lightning_logs/version_78/checkpoints/epoch=29-step=5610.ckpt')# 4层输入
# parser.add_argument('--checkpoint-path1', type=str, default='lightning_logs/version_89/checkpoints/epoch=9-step=1870.ckpt')# 4层输入
parser.add_argument('--segmentNetPath', type=str, default='lightning_logs/version_171/checkpoints/epoch=29-step=3360.ckpt')

# version_71 MobileNet 30
# version_76 ResNet50 30
# version_75 regNet 30
# version_77 ResNet18 30
# version_78 VggNet 30
# version_83 efficientNet 30
# version_85 denseNet121 30

# version_80 MobileNet 10
# version_81 ResNet18 10
# version_82 VggNet 10

# version_105 SegmentMobileNet 30
# version_104 SegmentVggNet 30
# version_101 SegmentResNet18 30
# version_100 SegmentNet(180 range range) 30

# version_112 SegmentNet(180 range range 4 input) 30

# version_140 SegmentMobileNet 30 augs


args = parser.parse_args()
overlay_img_fps = list(Path('../dataset/coco/val2017').glob('*.jpg'))
overlay_img_fps = None


# image_fps = list(Path('../dataset/classify/train/image').glob('*.png'))
# annotate_fps = list(Path('../dataset/classify/train/annotate').glob('*.json'))

# val_image_fps1 = list(Path('../dataset/classify/validation4/image').glob('*.png'))
# val_annotate_fps1 = list(Path('../dataset/classify/validation4/annotate').glob('*.json'))
#
# val_image_fps2 = list(Path('../dataset/validationDataset3/image').glob('*.png'))
# val_annotate_fps2 = list(Path('../dataset/validationDataset3/annotate').glob('*.json'))
#
# val_image_fps3 = list(Path('../dataset/validationDataset4_2/image').glob('*.png'))
# val_annotate_fps3 = list(Path('../dataset/validationDataset4_2/annotate').glob('*.json'))
#
# val_image_fps4 = list(Path('../dataset/validationDataset5/image').glob('*.png'))
# val_annotate_fps4 = list(Path('../dataset/validationDataset5/annotate').glob('*.json'))
val_image_fps_t1 = list(Path('../dataset/vrepImageVal/hole2/image').glob('*.png'))
val_annotate_fps_t1 = list(Path('../dataset/vrepImageVal/hole2/annotate').glob('*.json'))

val_image_fps_t2 = list(Path('../dataset/vrepImageVal/hole4/image').glob('*.png'))
val_annotate_fps_t2 = list(Path('../dataset/vrepImageVal/hole4/annotate').glob('*.json'))

val_image_fps_t3 = list(Path('../dataset/vrepImageVal/hole9/image').glob('*.png'))
val_annotate_fps_t3 = list(Path('../dataset/vrepImageVal/hole9/annotate').glob('*.json'))

val_image_fps_t4 = list(Path('../dataset/vrepImageVal/hole10/image').glob('*.png'))
val_annotate_fps_t4 = list(Path('../dataset/vrepImageVal/hole10/annotate').glob('*.json'))

val_image_fps_t5 = list(Path('../dataset/vrepImageVal/hole11/image').glob('*.png'))
val_annotate_fps_t5 = list(Path('../dataset/vrepImageVal/hole11/annotate').glob('*.json'))

val_image_fps_t6 = list(Path('../dataset/vrepImageVal/hole12/image').glob('*.png'))
val_annotate_fps_t6 = list(Path('../dataset/vrepImageVal/hole12/annotate').glob('*.json'))

val_image_fps_t7 = list(Path('../dataset/vrepImageVal/hole13/image').glob('*.png'))
val_annotate_fps_t7 = list(Path('../dataset/vrepImageVal/hole13/annotate').glob('*.json'))

val_image_fps1 = list(Path('../dataset/vrepImage/hole1/image').glob('*.png'))
val_annotate_fps1 = list(Path('../dataset/vrepImage/hole1/annotate').glob('*.json'))

val_image_fps2 = list(Path('../dataset/vrepImage/hole3/image').glob('*.png'))
val_annotate_fps2 = list(Path('../dataset/vrepImage/hole3/annotate').glob('*.json'))

val_image_fps3 = list(Path('../dataset/vrepImage/hole5/image').glob('*.png'))
val_annotate_fps3 = list(Path('../dataset/vrepImage/hole5/annotate').glob('*.json'))

val_image_fps4 = list(Path('../dataset/vrepImage/hole6/image').glob('*.png'))
val_annotate_fps4 = list(Path('../dataset/vrepImage/hole6/annotate').glob('*.json'))

val_image_fps5 = list(Path('../dataset/vrepImage/hole7/image').glob('*.png'))
val_annotate_fps5 = list(Path('../dataset/vrepImage/hole7/annotate').glob('*.json'))

val_image_fps6 = list(Path('../dataset/vrepImage/hole8/image').glob('*.png'))
val_annotate_fps6 = list(Path('../dataset/vrepImage/hole8/annotate').glob('*.json'))

val_image_fps7 = list(Path('../dataset/vrepImage/hole14/image').glob('*.png'))
val_annotate_fps7 = list(Path('../dataset/vrepImage/hole14/annotate').glob('*.json'))

# val_image_fps8 = list(Path('../dataset/classify/validation4/image').glob('*.png'))
# val_annotate_fps8 = list(Path('../dataset/classify/validation4/annotate').glob('*.json'))

val_image_fps8 = list(Path('../dataset/realDataset/image').glob('*.png'))
val_annotate_fps8 = list(Path('../dataset/realDataset/annotate').glob('*.json'))

val_image_fps9 = list(Path('../dataset/realDataset2/image').glob('*.png'))
val_annotate_fps9 = list(Path('../dataset/realDataset2/annotate').glob('*.json'))

val_image_fps10 = list(Path('../dataset/realDataset3/image').glob('*.png'))
val_annotate_fps10 = list(Path('../dataset/realDataset3/annotate').glob('*.json'))

val_imagepath_list=[
    # '../dataset/vrepImageVal/hole0/image',
    # '../dataset/vrepImage/hole1/image',
    '../dataset/vrepImage/hole3/image',
    '../dataset/vrepImage/hole5/image',
    '../dataset/vrepImage/hole6/image',
    '../dataset/vrepImage/hole7/image',
    '../dataset/vrepImage/hole8/image',
    '../dataset/vrepImage/hole14/image',
    # '../dataset/realDataset/image',
    # '../dataset/realDataset2/image',
    # '../dataset/realDataset3/image',
    # '../dataset/vrepImageVal/hole2/image',
    # '../dataset/vrepImageVal/hole4/image',
    # '../dataset/vrepImageVal/hole9/image',
    # '../dataset/vrepImageVal/hole10/image',
    # '../dataset/vrepImageVal/hole11/image',
    # '../dataset/vrepImageVal/hole12/image',
    # '../dataset/vrepImageVal/hole13/image',
]

val_annotatepath_list=[
    # '../dataset/vrepImageVal/hole0/annotate',
    # '../dataset/vrepImage/hole1/annotate',
    '../dataset/vrepImage/hole3/annotate',
    '../dataset/vrepImage/hole5/annotate',
    '../dataset/vrepImage/hole6/annotate',
    '../dataset/vrepImage/hole7/annotate',
    '../dataset/vrepImage/hole8/annotate',
    '../dataset/vrepImage/hole14/annotate',
    # '../dataset/realDataset/annotate',
    # '../dataset/realDataset2/annotate',
    # '../dataset/realDataset3/annotate',
    # '../dataset/vrepImageVal/hole2/annotate',
    # '../dataset/vrepImageVal/hole4/annotate',
    # '../dataset/vrepImageVal/hole9/annotate',
    # '../dataset/vrepImageVal/hole10/annotate',
    # '../dataset/vrepImageVal/hole11/annotate',
    # '../dataset/vrepImageVal/hole12/annotate',
    # '../dataset/vrepImageVal/hole13/annotate',
]

# val_imagepath_list = [val_image_fps1,val_image_fps2,val_image_fps3,val_image_fps4,val_image_fps5,val_image_fps6,val_image_fps7,val_image_fps8,val_image_fps9,val_image_fps10]
# val_annotatepath_list = [val_annotate_fps1,val_annotate_fps2,val_annotate_fps3,val_annotate_fps4,val_annotate_fps5,val_annotate_fps6,val_annotate_fps7,val_annotate_fps8,val_annotate_fps9,val_annotate_fps10]

# val_imagepath_list = [val_image_fps_t1,val_image_fps_t2,val_image_fps_t3,val_image_fps_t4,val_image_fps_t5,val_image_fps_t6,val_image_fps_t7]
# val_annotatepath_list = [val_annotate_fps_t1,val_annotate_fps_t2,val_annotate_fps_t3,val_annotate_fps_t4,val_annotate_fps_t5,val_annotate_fps_t6,val_annotate_fps_t7]

# val_imagepath_list = [val_image_fps7,val_image_fps8]
# val_annotatepath_list = [val_annotate_fps7,val_annotate_fps8]
# val_image_fps = list(Path('../dataset/realDataset/image2').glob('*.png'))
# val_annotate_fps = list(Path('../dataset/realDataset/annotate').glob('*.json'))
#
# val_image_fps8 = list(Path('../dataset/realDataset2/1/new').glob('*.png'))
# val_annotate_fps8 = list(Path('../dataset/classify/validation4/annotate').glob('*.json'))
# val_imagepath_list = [val_image_fps8]
# val_annotatepath_list = [val_annotate_fps8]

# train_kwargs = dict(augs=False, overlay_image_fps=overlay_img_fps,inputLayersNum=args.inputLayersNum)
# data_valid = VggDataset(val_image_fps, val_annotate_fps, **train_kwargs)


# loader_kwargs = dict(batch_size=8, num_workers=5, persistent_workers=True)
# loader_train = torch.utils.data.DataLoader(dataset=data_train, shuffle=True, drop_last=True, **loader_kwargs)
# loader_valid = torch.utils.data.DataLoader(dataset=data_valid, **loader_kwargs)
# learner = pl.Trainer(
#     accelerator='gpu',
#     callbacks=[
#         pl.callbacks.ModelCheckpoint(monitor='loss_val', save_weights_only=True),
#         pl.callbacks.EarlyStopping(monitor='loss_val', patience=10)
#     ]
# )
#
# model = Model(loss=args.loss)
# if __name__ == '__main__':
#     multiprocessing.freeze_support()  # 这行在 Windows 上是必需的
#     learner.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_valid, ckpt_path=args.checkpoint_path)

# # 图像归一化
# transform_GY = transforms.ToTensor()  # 将PIL.Image转化为tensor，即归一化。注：shape 会从(H，W，C)变成(C，H，W)
# # 图像标准化
# transform_BZ = transforms.Normalize(
#     mean=[0.5, 0.5, 0.5],  # 取决于数据集
#     std=[0.5, 0.5, 0.5]
# )
# # transform_compose
# transform_compose = transforms.Compose([
#     # 先归一化再标准化
#     transform_GY,
#     transform_BZ
# ])
if args.inputLayersNum==4:
    model = ClassificationModel.load_from_checkpoint(args.checkpoint_path2, loss=args.loss, inputLayersNum = args.inputLayersNum, networkName= args.networkName, segment_checkpoint_path = args.segmentNetPath, augs=args.augs)
elif args.inputLayersNum==3:
    model = ClassificationModel.load_from_checkpoint(args.checkpoint_path2, loss=args.loss, inputLayersNum = args.inputLayersNum, networkName= args.networkName, segment_checkpoint_path = args.segmentNetPath)
else:
    raise ValueError
model.eval()
successRateList = []
for n in range(len(val_imagepath_list)):
    val_image_fps = list(Path(val_imagepath_list[n]).glob('*.png'))
    val_annotate_fps = list(Path(val_annotatepath_list[n]).glob('*.json'))
    train_kwargs = dict(augs=False, overlay_image_fps=overlay_img_fps, inputLayersNum=args.inputLayersNum)
    data_valid = VggDataset(val_image_fps, val_annotate_fps, **train_kwargs)
    successRate = []
    for i in range(len(args.image_num)):
        successNum = 0
        for idx in range(args.image_num[i]):
            idx = idx + np.sum(args.image_num[0:i])
            img, stateIdx, rep = data_valid.get(int(idx))
            if args.inputLayersNum == 4:
                img_tensor, _ = data_valid.normalize(img, stateIdx, rep)
            elif args.inputLayersNum==3:
                img_tensor, _ = data_valid.normalize(img, stateIdx)
            else:
                raise ValueError
            # img_tensor = torch.from_numpy(img).permute(2, 1, 0)/255 - 0.5
            # img_tensor = transform_compose(img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0).to(torch.float32)
            img_out = model.forward(img_tensor.to('cuda'))
            # mask = model.getSegment(img_tensor.to('cuda'))
            # mask = mask.cpu().detach().numpy()

            result = img_out.cpu().detach().numpy()

            print('result:', result)
            print('stateIdx:', stateIdx)

            a = np.where(result[0] == np.max(result[0]))
            b = np.where(stateIdx == np.max(stateIdx))
            if a[0] == b[0]:
                successNum += 1
            else:
                print('Classification error')

                # plt.figure()
                # plt.imshow(rep[0])
                # plt.figure()
                # plt.imshow(mask[0][0])
                # plt.figure()
                # plt.imshow(mask[0][1])
                # plt.figure()
                # plt.imshow(img)
                # plt.show()

            # plt.figure()
            # plt.imshow(img)
            # plt.show()
        successRate.append(successNum/args.image_num[i])

    print(successRate)
    successRateList.append(successRate)
print(successRateList)
np.save('../../results/version_218_successRateList.npy',successRateList)


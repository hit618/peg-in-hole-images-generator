import argparse
import multiprocessing
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from DualArmAsm.imageTrainer.makeDataset.dataset import SynthDataset
from model import Model
from pathlib import Path
from DualArmAsm.imageTrainer.makeDataset.dataset import SynthDataset

parser = argparse.ArgumentParser()
parser.add_argument('--loss', choices=['mse', 'bce'], default='mse')
# parser.add_argument('--checkpoint-path', type=str, default='lightning_logs/version_34/checkpoints/epoch=47-step=5376.ckpt')
parser.add_argument('--checkpoint-path', type=str, default='lightning_logs/version_188/checkpoints/epoch=29-step=7860.ckpt')

args = parser.parse_args()

overlay_img_fps = list(Path('../dataset/coco/val2017').glob('*.jpg'))
# image_fps = list(Path('test_image').glob('*.png'))
# annotate_fps = list(Path('../dataset/annotate').glob('*.json'))
#
# val_image_fps = list(Path('../dataset/valImage').glob('*.png'))
# val_annotate_fps = list(Path('../dataset/valAnnotate').glob('*.json'))

image_fps = list(Path('../dataset/classify/train/image').glob('*.png'))
annotate_fps = list(Path('../dataset/classify/train/annotate').glob('*.json'))

# val_image_fps = list(Path('../dataset/classify/validation/image').glob('*.png'))
# val_annotate_fps = list(Path('../dataset/classify/validation/annotate').glob('*.json'))

# val_image_fps = list(Path('../dataset/validationDataset4/image').glob('*.png'))
# val_annotate_fps = list(Path('../dataset/validationDataset4/annotate').glob('*.json'))

val_image_fps = list(Path('../dataset/realDataset/image').glob('*.png'))
val_annotate_fps = list(Path('../dataset/realDataset/annotate').glob('*.json'))

train_kwargs = dict(augs=False, overlay_image_fps=overlay_img_fps)
data_train = SynthDataset(image_fps,annotate_fps, **train_kwargs)
data_valid = SynthDataset(val_image_fps,val_annotate_fps, **train_kwargs)


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

# 图像归一化
transform_GY = transforms.ToTensor()  # 将PIL.Image转化为tensor，即归一化。注：shape 会从(H，W，C)变成(C，H，W)
# 图像标准化
transform_BZ = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],  # 取决于数据集
    std=[0.5, 0.5, 0.5]
)
# transform_compose
transform_compose = transforms.Compose([
    # 先归一化再标准化
    transform_GY,
    transform_BZ
])

model = Model.load_from_checkpoint(args.checkpoint_path, loss=args.loss)
model.eval()
for idx in range(50,70):
    img, rep, peg_pos, hole_pos = data_valid.get(idx)
    img_tensor, _ = data_valid.normalize(img)
    # img_tensor = torch.from_numpy(img).permute(2, 1, 0)/255 - 0.5
    # img_tensor = transform_compose(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(torch.float32)
    img_out = model.forward(img_tensor.to('cuda'))
    img_out = torch.squeeze(img_out).cpu().detach().numpy()

    position1 =  np.where(img_out[0]==np.max(img_out[0])) #hole position
    position2 =  np.where(img_out[1]==np.max(img_out[1])) #peg position
    img[position1[0][0]][position1[1][0]] = [1,1,1]
    img[position2[0][0]][position2[1][0]] = [1,1,1]

    plt.figure()
    plt.imshow(img_out[0])
    plt.figure()
    plt.imshow(img_out[1])
    plt.figure()
    plt.imshow(img)
    # plt.figure()
    # plt.imshow(rep[0])
    # plt.figure()
    # plt.imshow(rep[1])
    plt.show()


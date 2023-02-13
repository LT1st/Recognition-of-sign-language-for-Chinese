import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from models.Conv3D import r2plus1d_18
from dataset_sign_clip import Sign_Isolated
from train import train_epoch
from collections import OrderedDict


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


# Path setting
data_path = "../../Dataset/frames/"  # 训练路径
label_train_path = "../../Dataset/rawData/train.csv"  # 训练的标签

# writer = SummaryWriter(sum_path)        # 使用了tensorboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 226  # 最终的分类目标数
epochs = 16  # 训练轮数
batch_size = 4
learning_rate = 1e-3  # 1e-3 Train 1e-4 Finetune
weight_decay = 1e-4  # 1e-4
log_interval = 4   # 注册间隔
sample_size = 64
sample_duration = 24
attention = False
drop_p = 0.0


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    # 数据相关
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = Sign_Isolated(data_path=data_path, label_path=label_train_path, frames=sample_duration,
                              num_classes=num_classes, train=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    # 模型相关

    # 这里载入的模型虽然也是预训练，但是是在其他通用数据集上完成的
    model = r2plus1d_18(pretrained=True, num_classes=num_classes)
    # 载入训练模型，这里的训练模型来源于暂停训练后保留的模型
    # checkpoint = torch.load('models/slr_resnet2d+1.pth')
    # new_state_dict = OrderedDict()

    # for k, v in checkpoint.items():
    #     print(k[7:])
    #     name = k[7:]  # 对于训练的模型，我们只要权重部分(模型结构部分不要)
    #     new_state_dict[name] = v

    # model.load_state_dict(new_state_dict)  # 重新载入含有训练权重的模型

    # # 这一段代码不明所以，fc层早就被替换过了，而且num_classes=226为什么不在上面构造函数的时候指定呢?
    # model.fc1 = nn.Linear(model.fc1.in_features, num_classes)

    model = model.to(device)

    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.SGD(  # 显存不足,被迫改优化器
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)

    # Start training
    for epoch in range(epochs):
        print('lr: ', get_lr(optimizer))
        train_epoch(model, criterion, optimizer, train_loader,
                    device, epoch, log_interval, None)

    dataiter = iter(train_loader)
    images = next(dataiter)['data']
    images = images.to(device)


# Train with 3DCNN
if __name__ == '__main__':
    main()

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
from trainLoop import train_epoch
from testLoop import test_epoch


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
data_path = "./DataPreper/SelfTrain/"  # 训练路径
test_path = "./DataPreper/SelfTest/"
label_train_path = "../Dataset/train.csv"  # 训练的标签
label_test_path = "../Dataset/test.csv"  # 训练的标签
# writer = SummaryWriter(sum_path)        # 使用了tensorboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 226  # 最终的分类目标数
start_model = 3
epochs = 4  # 训练轮数
batch_size = 6
learning_rate = 0.002  # 0.003-0.001 Train 0.0004-0.0001 Finetune
weight_decay = 1e-4  # 1e-4
log_interval = 100   # 注册间隔
sample_size = 96
sample_duration = 16
attention = False


def train():
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
    model = r2plus1d_18(pretrained=False, num_classes=num_classes)

    # 载入训练模型，这里的训练模型来源于暂停训练后保留的模型
    modelName = f"./models/3dcnn_{start_model}.pth"
    print(f"载入模型: {modelName}")
    checkpoint = torch.load(modelName)
    model.load_state_dict(checkpoint)  # 重新载入含有训练权重的模型

    model = model.to(device)

    loss_fn = LabelSmoothingCrossEntropy()
    optimizer = optim.SGD(  # 显存不足,被迫改优化器
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Start training
    for epoch in range(start_model, start_model+epochs):
        train_epoch(model, loss_fn, optimizer, train_loader,
                    device, epoch, log_interval, None)
        torch.save(model.state_dict(), f"./models/3dcnn_{epoch+1}.pth")


def test(startIndex, endIndex):
    # 数据相关
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    test_set = Sign_Isolated(data_path=test_path, label_path=label_test_path, frames=sample_duration,
                             num_classes=num_classes, train=False, transform=transform)

    test_loader = DataLoader(
        test_set, batch_size=2, shuffle=False, num_workers=12, pin_memory=True)
    # 模型相关

    # 这里载入的模型虽然也是预训练，但是是在其他通用数据集上完成的
    model = r2plus1d_18(pretrained=False, num_classes=num_classes)
    for index in range(startIndex, endIndex+1):
        # 载入训练模型，这里的训练模型来源于暂停训练后保留的模型
        modelName = f"./models/3dcnn_{index}.pth"
        print(f"测试模型: {modelName}")
        checkpoint = torch.load(modelName)
        model.load_state_dict(checkpoint)  # 重新载入含有训练权重的模型
        model = model.to(device)
        test_epoch(model, test_loader, device, None)


# Train with 3DCNN
if __name__ == '__main__':
    # train()
    test(3, 7)

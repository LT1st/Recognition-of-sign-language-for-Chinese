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

from SignDataset import SignDataset
from trainLoop import train_epoch
from testLoop import test_epoch


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1):
        classNum = pred.shape[1]
        logProbs = F.log_softmax(pred, dim=1)
        oneHotTarget = F.one_hot(target, classNum)
        oneHotTarget = torch.clamp(
            oneHotTarget.float(), min=smoothing/(classNum-1), max=1-smoothing)
        loss = - torch.sum(oneHotTarget*logProbs, 1)
        return loss.mean()


# Path setting
dataPath = "./DataPreper/Train/"  # 训练路径
testPath = "./DataPreper/Test/"
modelPath = "./models/"
# writer = SummaryWriter(sum_path)        # 使用了tensorboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 500  # 最终的分类目标数
dataSize = 240
start_model = 20
epochs = 1  # 训练轮数
batch_size = 6
learning_rate = 0.0001  # 0.003-0.001 Train 0.0004-0.0001 Finetune
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
    trainSet = SignDataset(dataPath=dataPath, frames=sample_duration,
                           dataSize=dataSize, train=True, transform=transform)

    trainLoader = DataLoader(
        trainSet, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    # 模型相关

    # 这里载入的模型虽然也是预训练，但是是在其他通用数据集上完成的
    model = r2plus1d_18(pretrained=False, num_classes=num_classes)

    # 载入训练模型，这里的训练模型来源于暂停训练后保留的模型
    modelName = os.path.join(modelPath, f"3dcnn_{start_model}.pth")
    print(f"载入模型: {modelName}")
    checkpoint = torch.load(modelName)
    model.load_state_dict(checkpoint)  # 重新载入含有训练权重的模型

    model = model.to(device)

    loss_fn = LabelSmoothingCrossEntropy()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(  # 显存不足,被迫改优化器
        model.parameters(), lr=learning_rate, momentum=0.85, weight_decay=weight_decay)

    # Start training
    for epoch in range(start_model, start_model+epochs):
        train_epoch(model, loss_fn, optimizer, trainLoader,
                    device, epoch, log_interval, None)
        torch.save(model.state_dict(), os.path.join(
            modelPath, f"3dcnn_{epoch+1}.pth"))


def test(startIndex, endIndex):
    # 数据相关
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    testSet = SignDataset(dataPath=testPath, frames=sample_duration,
                          dataSize=10, train=False, transform=transform)

    testLoader = DataLoader(
        testSet, batch_size=3, shuffle=True, num_workers=4, pin_memory=True)
    # 模型相关

    # 这里载入的模型虽然也是预训练，但是是在其他通用数据集上完成的
    model = r2plus1d_18(pretrained=False, num_classes=num_classes)
    for index in range(startIndex, endIndex+1):
        # 载入训练模型，这里的训练模型来源于暂停训练后保留的模型
        modelName = os.path.join(modelPath, f"3dcnn_{index}.pth")
        print(f"测试模型: {modelName}")
        checkpoint = torch.load(modelName)
        model.load_state_dict(checkpoint)  # 重新载入含有训练权重的模型
        model = model.to(device)
        test_epoch(model, testLoader, device, None)


# Train with 3DCNN
if __name__ == '__main__':
    # train()
    test(19, 20)

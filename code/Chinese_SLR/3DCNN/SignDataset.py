import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import math
import numpy as np


class SignDataset(Dataset):
    def __init__(self, dataPath,  frames=16, dataSize=240, train=True, transform=None) -> None:
        super(SignDataset).__init__()
        self.dataPath = dataPath
        self.frames = frames
        self.train = train
        self.transform = transform
        self.folderDataSzie = dataSize

    def __len__(self):
        return 500 * self.folderDataSzie

    def __getitem__(self, index):
        label = math.floor(index/self.folderDataSzie)  # 获得标签
        labelFolder = os.path.join(self.dataPath, "{:03d}".format(label))
        imagesFolder = os.path.join(
            labelFolder, os.listdir(labelFolder)[index-label*self.folderDataSzie])

        images = self.getImages(imagesFolder)   # 获得图像
        return {'data': images, 'label': torch.LongTensor([label])}  # 返回图像和标签

    def getImages(self, dir: str):
        """
        从文件夹内获取三维的图像序列，如果在训练模式下，还会对数据集进行扩充
        """
        images = []
        # 获取视频帧序列
        imagesPathlist = self.getImagesPath(len(os.listdir(dir)))
        angle = (random.random() - 0.5) * 10        # 旋转扩充数据集

        for i in imagesPathlist:
            # 获得图像
            image = Image.open(os.path.join(dir, '{:06d}.jpg').format(i))
            if self.train:
                image = transforms.functional.rotate(image, angle)
            if self.transform is not None:
                image = self.transform(image)
            # 加入图像集
            images.append(image)

        images = torch.stack(images, dim=0)
        # 调整为3dCNN的形式
        images = images.permute(1, 0, 2, 3)
        return images

    def getImagesPath(self, videoLen: int):
        """
        如果视频长度大于采样时长，则随机选择一个合适的起始点，并生成从随机起始点到结束的等分数据段
        如果视频长度不大于采样时长，则生成整个视频的帧索引，并将其循环直到达到采样时长。
        返回一个shape为(sample_duration,)的帧索引数组。
        """
        frameIndices = None
        if videoLen > self.frames:
            indexLen = math.floor(videoLen/self.frames)  # 获得间隔长度
            randomStart = random.randint(      # 选择随机起始点
                0, videoLen - self.frames*indexLen)
            frameIndices = np.arange(
                randomStart, randomStart + self.frames*indexLen, indexLen)
        else:
            frameIndices = np.arange(videoLen)
            while frameIndices.shape[0] < self.frames:
                frameIndices = np.concatenate(
                    (frameIndices, np.arange(videoLen)), axis=0)
            frameIndices = frameIndices[:self.frames]
        assert frameIndices.shape[0] == self.frames
        return frameIndices+1  # 这份数据集从1开始，所以整体+1


# 测试数据集工作正常
if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize([24, 24]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = SignDataset('./DataPreper/Train', transform=transform)
    for i in range(30000, 100000, 150):
        dataset.__getitem__(i)

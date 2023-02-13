import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np

"""
Implementation of Sign Language Dataset
"""


class Sign_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=226, train=True, transform=None, test_clips=5):
        super(Sign_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.test_clips = test_clips
        self.sample_names = []
        self.labels = []
        self.data_folder = []   # 存放着图片数据路径，根据标记文件生成
        label_file = open(label_path, 'r', encoding='utf-8')    # 读取标签
        for line in label_file.readlines():     # 根据标签，生成需要读取图形的路径
            line = line.strip()
            line = line.split(',')

            self.sample_names.append(line[0])
            self.data_folder.append(os.path.join(data_path, line[0]))
            self.labels.append(int(line[1]))

    def frame_indices_tranform(self, video_length, sample_duration):
        """
        如果视频长度大于采样时长，则随机选择一个起始点，并生成从随机起始点到随机起始点加采样时长的帧索引。
        如果视频长度不大于采样时长，则生成整个视频的帧索引，并将其循环直到达到采样时长。
        返回一个shape为(sample_duration,)的帧索引数组。
        """
        if video_length > sample_duration:
            random_start = random.randint(0, video_length - sample_duration)
            frame_indices = np.arange(
                random_start, random_start + sample_duration) + 1
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate(
                    (frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration] + 1
        assert frame_indices.shape[0] == sample_duration
        return frame_indices

    def frame_indices_tranform_test(self, video_length, sample_duration, clip_no=0):
        if video_length > sample_duration:
            start = (video_length -
                     sample_duration) // (self.test_clips - 1) * clip_no
            frame_indices = np.arange(start, start + sample_duration) + 1
        elif video_length == sample_duration:
            frame_indices = np.arange(sample_duration) + 1
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate(
                    (frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration] + 1

        return frame_indices

    def random_crop_paras(self, input_size, output_size):
        # 生成随机裁剪参数
        diff = input_size - output_size
        i = random.randint(0, diff)
        j = random.randint(0, diff)
        return i, j, i+output_size, j+output_size

    def read_images(self, folder_path, clip_no=0):
        """
        从文件夹内获取三维的图像序列，如果在训练模式下，还会对数据集进行扩充
        PS: 翻转镜像等功能不会造成手语识别变得更加迷惑吗，或许有的手语语义是存在左右关系的,也许这一部分代码功能应该被去除
        """
        # assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        if self.train:
            index_list = self.frame_indices_tranform(   # 获取视频帧序列
                len(os.listdir(folder_path)), self.frames)
            flip_rand = random.random()                 # 生成随机数判断是否要翻转以扩充数据集
            angle = (random.random() - 0.5) * 10        # 旋转扩充数据集
            crop_box = self.random_crop_paras(256, 224)
        else:
            index_list = self.frame_indices_tranform_test(
                len(os.listdir(folder_path)), self.frames, clip_no)

        # for i in range(self.frames):
        for i in index_list:
            image = Image.open(os.path.join(
                folder_path, '{:d}.jpg').format(i-1))
            if self.train:
                # if flip_rand > 0.5:
                #     image = ImageOps.mirror(image)
                image = transforms.functional.rotate(image, angle)
                image = image.crop(crop_box)
                assert image.size[0] == 224
            else:
                crop_box = (16, 16, 240, 240)
                image = image.crop(crop_box)
                # assert image.size[0] == 224
            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        return images

    def __len__(self):
        return len(self.data_folder)

    def __getitem__(self, idx):
        selected_folder = self.data_folder[idx]  # 获取数据目录
        if self.train:
            images = self.read_images(selected_folder)  # 这个函数获取三维的图片序列数据
        else:
            images = []
            for i in range(self.test_clips):
                images.append(self.read_images(selected_folder, i))
            images = torch.stack(images, dim=0)
            # M, T, C, H, W
        label = torch.LongTensor([self.labels[idx]])
        # print(images.size(), ', ', label.size())
        return {'data': images, 'label': label}  # 返回图像和标签

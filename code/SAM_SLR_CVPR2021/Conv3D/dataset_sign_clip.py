import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np

"""
用于数据集读取、划分
"""
class Sign_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=226, train=True, transform=None, test_clips=5):
        """_summary_

        :param data_path: _description_
        :type data_path: _type_
        :param label_path: _description_
        :type label_path: _type_
        :param frames: 帧数, defaults to 16
        :type frames: int, optional
        :param num_classes: 词汇类别数, defaults to 226
        :type num_classes: int, optional
        :param train: 用于训练, defaults to True
        :type train: bool, optional
        :param transform: 数据增强, defaults to None
        :type transform: _type_, optional
        :param test_clips: 测试集划分量, defaults to 5
        :type test_clips: int, optional
        """
        super(Sign_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.test_clips = test_clips
        self.sample_names = []          # 标签名
        self.labels = []                # 标签对应数字
        self.data_folder = []           # 文件夹
        label_file = open(label_path, 'r', encoding='utf-8')
        for line in label_file.readlines():
            line = line.strip()
            line = line.split(',')

            self.sample_names.append(line[0])
            self.data_folder.append(os.path.join(data_path, line[0]))
            self.labels.append(int(line[1]))

    def frame_indices_tranform(self, video_length, sample_duration):
        """ 视频转换帧，并长度标准化

        :param video_length: 视频帧长度
        :type video_length: int
        :param sample_duration: 采样长度，既目标帧长度，论文150
        :type sample_duration: int
        :return: _description_
        :rtype: _type_
        """
        if video_length > sample_duration:  #此视频太长
            # 随机取一部分
            random_start = random.randint(0, video_length - sample_duration)
            frame_indices = np.arange(random_start, random_start + sample_duration) + 1
        else:   #此视频不太长
            frame_indices = np.arange(video_length)
            # 视频不够长就重复帧来补齐
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration] + 1
        assert frame_indices.shape[0] == sample_duration
        return frame_indices

    def frame_indices_tranform_test(self, video_length, sample_duration, clip_no=0):
        """视频长度标准化

        :param video_length: 视频帧长度
        :type video_length: int
        :param sample_duration: 采样长度，既目标帧长度，论文150
        :type sample_duration: int
        :param clip_no: 测试集划分量, defaults to 0 ???
        :type clip_no: int, optional
        :return: _description_
        :rtype: _type_
        """
        if video_length > sample_duration:
            start = (video_length - sample_duration) // (self.test_clips - 1) * clip_no
            frame_indices = np.arange(start, start + sample_duration) + 1
        elif video_length == sample_duration:
            frame_indices = np.arange(sample_duration) + 1
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration] + 1

        return frame_indices

    def random_crop_paras(self, input_size, output_size):
        """随机裁切

        :return: ROI参数
        :rtype: int, int, int, int
        """
        diff = input_size - output_size
        i = random.randint(0, diff)
        j = random.randint(0, diff)
        return i, j, i+output_size, j+output_size

    def read_images(self, folder_path, clip_no=0):
        """读取一个视频的图？

        :param folder_path: 文件
        :type folder_path: _type_
        :param clip_no: 视频编号, defaults to 0
        :type clip_no: int, optional
        :return: _description_
        :rtype: _type_
        """
        # assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []

        # 训练集
        if self.train:
            # 这里数据是按照图像储存的
            index_list = self.frame_indices_tranform(len(os.listdir(folder_path)), self.frames)
            flip_rand = random.random() # 翻转
            angle = (random.random() - 0.5) * 10    # 角度
            crop_box = self.random_crop_paras(256, 224) # 随机裁切
        # 测试集不做增强
        else:
            index_list = self.frame_indices_tranform_test(len(os.listdir(folder_path)), self.frames, clip_no)
        
        # for i in range(self.frames):
        for i in index_list:
            image = Image.open(os.path.join(folder_path, '{:04d}.jpg').format(i))
            if self.train:
                # 翻转阈值
                if flip_rand > 0.5:
                    image = ImageOps.mirror(image)
                image = transforms.functional.rotate(image, angle)
                image = image.crop(crop_box)
                assert image.size[0] == 224
            # 测试集固定数据
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
        selected_folder = self.data_folder[idx]
        if self.train:
            images = self.read_images(selected_folder)

        # 测试集从特定划分直接读取
        else:
            images = []
            for i in range(self.test_clips):
                images.append(self.read_images(selected_folder, i))
            images = torch.stack(images, dim=0)
            # M, T, C, H, W

        label = torch.LongTensor([self.labels[idx]])
        # print(images.size(), ', ', label.size())
        return {'data': images, 'label': label}


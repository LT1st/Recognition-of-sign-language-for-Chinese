import os
import torch
import torch.nn.parallel
import torch.nn.functional as f
import torchvision.transforms as transforms
from pose_hrnet import get_pose_net
from collections import OrderedDict
from config import cfg

import numpy as np
import cv2

from utils import pose_process
from natsort import natsorted

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./runs")

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

index_mirror = np.concatenate([
    [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
    [21, 22, 23, 18, 19, 20],
    np.arange(40, 23, -1), np.arange(50, 40, -1),
    np.arange(51, 55), np.arange(59, 54, -1),
    [69, 68, 67, 66, 71, 70], [63, 62, 61, 60, 65, 64],
    np.arange(78, 71, -1), np.arange(83, 78, -1),
    [88, 87, 86, 85, 84, 91, 90, 89],
    np.arange(113, 134), np.arange(92, 113)
]) - 1
assert (index_mirror.shape[0] == 133)

multi_scales = [512, 640]


def norm_numpy_totensor(img):
    """
    对一个输入的四维numpy数组进行归一化并转化为 PyTorch 的 Tensor 类型。
    """
    img = img.astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]
    return torch.from_numpy(img).permute(0, 3, 1, 2)


def stack_flip(img):
    img_flip = cv2.flip(img, 1)
    return np.stack([img, img_flip], axis=0)


def merge_hm(hms_list):
    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1, :, :, :] = torch.flip(hms[1, index_mirror, :, :], [2])

    hm = torch.cat(hms_list, dim=0)
    # print(hm.size(0))
    hm = torch.mean(hms, dim=0)
    return hm


def main(inputPath="../../Dataset/rawData/train/", outputPath="../../Dataset/wholepose/", modelPath="../../Models/"):
    with torch.no_grad():
        # 加载模型，并导入预设参数，同时设定为评估模式
        cfg.merge_from_file('wholebody_w48_384x288.yaml')
        newmodel = get_pose_net(cfg, is_train=False)
        checkpoint = torch.load(
            modelPath + 'hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth')
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'backbone.' in k:
                name = k[9:]  # remove module.
            if 'keypoint_head.' in k:
                name = k[14:]  # remove module.
            new_state_dict[name] = v
        newmodel.load_state_dict(new_state_dict)
        newmodel.cuda().eval()

        # 对指定目录进行遍历，同时自动排除深度视频，获得RGB视频路径
        input_path = inputPath
        paths = []
        names = []
        for root, _, fnames in natsorted(os.walk(input_path)):
            for fname in natsorted(fnames):
                path1 = os.path.join(root, fname)
                if 'depth' in fname:
                    continue
                paths.append(path1)
                names.append(fname)

        # 对获得到的系列文件进行处理
        for i, path in enumerate(paths):
            output_npy = outputPath+'{}.npy'.format(
                names[i])
            # 如果已经存在输出文件了，跳过
            if os.path.exists(output_npy):
                continue
            # 读取该视频，并获取其宽和高。
            cap = cv2.VideoCapture(path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            output_list = []

            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break

                frame_height, frame_width = img.shape[:2]
                img = cv2.flip(img, flipCode=1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out = []
                for scale in multi_scales:
                    if scale != 512:
                        img_temp = cv2.resize(img, (scale, scale))
                    else:
                        img_temp = img
                    img_temp = stack_flip(img_temp)
                    img_temp = norm_numpy_totensor(img_temp).cuda()
                    hms = newmodel(img_temp)

                    if scale != 512:
                        out.append(f.interpolate(
                            hms, (frame_width // 4, frame_height // 4), mode='bilinear'))
                    else:
                        out.append(hms)

                out = merge_hm(out)

                result = out.reshape((133, -1))
                result = torch.argmax(result, dim=1)

                result = result.cpu().numpy().squeeze()

                y = result // (frame_width // 4)
                x = result % (frame_width // 4)
                pred = np.zeros((133, 3), dtype=np.float32)
                pred[:, 0] = x
                pred[:, 1] = y

                hm = out.cpu().numpy().reshape((133, frame_height//4, frame_height//4))

                pred = pose_process(pred, hm)
                pred[:, :2] *= 4.0

                assert pred.shape == (133, 3)

                output_list.append(pred)

            output_list = np.array(output_list)

            np.save(output_npy, output_list)
            cap.release()


if __name__ == '__main__':
    main()

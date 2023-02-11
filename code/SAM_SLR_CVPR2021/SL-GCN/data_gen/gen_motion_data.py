import os
import numpy as np
from numpy.lib.format import open_memmap

# 不同时间来计算动作
sets = {
    'train', 'val', 'test'

}

datasets = {
    'sign/27_2'
}

parts = {
    'joint', 'bone'
}
from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        for part in parts:
            print(dataset, set, part)
            data = np.load('../data/{}/{}_data_{}.npy'.format(dataset, set, part))
            N, C, T, V, M = data.shape
            print(data.shape)
            fp_sp = open_memmap(
                '../data/{}/{}_data_{}_motion.npy'.format(dataset, set, part),
                dtype='float32',
                mode='w+',
                shape=(N, C, T, V, M))
            for t in tqdm(range(T - 1)):
                fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :] # 前一个帧减去下一个帧，得到动作
            fp_sp[:, :, T - 1, :, :] = 0

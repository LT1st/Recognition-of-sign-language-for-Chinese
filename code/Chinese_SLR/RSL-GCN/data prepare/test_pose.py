import os

import numpy as np
# input_data = np.load("npy3/signer0_sample1_color.mp4.npy")
# input_data = np.load("npy3/P01_01_01_1.npy")
# print(input_data.shape)
# data = input_data.reshape(1,-1)
# print(data.shape)
# print(data)
# # np.savetxt("test1.txt",data,delimiter=',')
# np.savetxt("test.txt",data,delimiter=',')
from natsort import natsorted
input_path = r'F:\SLRdataset\train'
paths = []
names = []
for root, dirs, fnames in natsorted(os.walk(input_path)):
    for dir in dirs:
        rootName = os.path.join(root, dir)
        for root_, dir_, _ in natsorted(os.walk(rootName)):
            sum=0
            for dir in dir_:
                sum+=1
                if sum>5:
                    break
                rootName = os.path.join(root_, dir)
                paths.append(rootName)
                names.append(dir)
            break
print(names)



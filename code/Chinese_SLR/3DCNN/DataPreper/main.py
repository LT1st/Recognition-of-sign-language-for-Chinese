import cv2
import numpy as np
from tqdm import tqdm
import os


# def crop(image, center, radius, size=512):
#     scale = 1.3
#     radius_crop = (radius * scale).astype(np.int32)
#     center_crop = (center).astype(np.int32)

#     rect = (max(0, (center_crop-radius_crop)[0]), max(0, (center_crop-radius_crop)[1]),
#             min(512, (center_crop+radius_crop)[0]), min(512, (center_crop+radius_crop)[1]))

#     image = image[rect[1]:rect[3], rect[0]:rect[2], :]

#     if image.shape[0] < image.shape[1]:
#         top = abs(image.shape[0] - image.shape[1]) // 2
#         bottom = abs(image.shape[0] - image.shape[1]) - top
#         image = cv2.copyMakeBorder(
#             image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
#     elif image.shape[0] > image.shape[1]:
#         left = abs(image.shape[0] - image.shape[1]) // 2
#         right = abs(image.shape[0] - image.shape[1]) - left
#         image = cv2.copyMakeBorder(
#             image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
#     return image


# selected_joints = np.concatenate(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                                   [91, 95, 96, 99, 100, 103, 104, 107, 108, 111], [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0)
# folder = './data/Validation/val'  # 'train', 'test'
# npy_folder = './data/npy3'  # 'train_npy/npy3', 'test_npy/npy3'
# out_folder = './data/frames'  # 'train_frames' 'test_frames'
def checkDir(dirName: str):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)


def video2images(inputPath: str, outputPath: str, interval=1):
    for videoName in tqdm(os.listdir(inputPath)):
        videoPath = inputPath + videoName
        name, _ = videoName.split('.')
        savePath = outputPath + name + "/"
        checkDir(savePath)

        index = 0
        cap = cv2.VideoCapture(videoPath)

        while (cap.isOpened()):
            success, frame = cap.read()
            if (not success):
                break
            frame = cv2.resize(frame, (128, 128))
            cv2.imwrite(savePath + f"{index}.jpg", frame)
            index += 1


if __name__ == "__main__":
    checkDir("SelfTrain")
    checkDir("SelfTest")
    # video2images("../../Dataset/train/", "./SelfTrain/")
    video2images("../../Dataset/test/", "./SelfTest/")

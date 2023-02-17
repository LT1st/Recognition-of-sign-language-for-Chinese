import cv2
import numpy as np
from tqdm import tqdm
import os


def checkDir(dirName: str):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)


def video2images(videoPath: str, outputPath: str):
    checkDir(outputPath)
    cap: cv2.VideoCapture = cv2.VideoCapture(videoPath)

    index = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        if (not success):
            break
        frame = cv2.resize(frame, (256, 256))
        cv2.imwrite(os.path.join(
            outputPath, "{:06d}.jpg".format(index)), frame)
        index += 1


def camera2images(outputPath: str):
    checkDir(outputPath)
    cap: cv2.VideoCapture = cv2.VideoCapture(0)  # 获得摄像头
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
    print(f"cameraWidth: {width}, cameraHeight: {height}")

    index = 0
    working = False
    while (cap.isOpened()):
        success, frame = cap.read()
        frame = frame[:, int((width-height)/2):int((width+height)/2)]
        cv2.putText(
            frame, f"Identification = {working}", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 3)

        cv2.imshow('camera', frame)
        key = cv2.waitKey(1)
        if (not success or key == ord('q')):
            break

        if (working):  # 如果开启录制，那么保存图片
            cv2.imwrite(os.path.join(
                outputPath, "{:06d}.jpg".format(index)), frame)
            index += 1
        else:   # 反之，停止保存并开始计算
            pass

    cap.release()


if __name__ == "__main__":
    video2images("/home/charain/下载/352432.mp4", "./input")

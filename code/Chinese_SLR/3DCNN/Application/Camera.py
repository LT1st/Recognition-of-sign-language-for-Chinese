import cv2
import os
import numpy as np
from Deduction import Deduction
import json


def checkDir(dirName: str):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)


def clearDir(dirName: str):
    if os.path.isdir(dirName):
        os.system(f"rm {dirName}/*")


class Camera:
    def __init__(self, model: Deduction, tmpFolder="./tmp/") -> None:
        self.model = model
        self.tmpFolder = tmpFolder
        self.cap = cv2.VideoCapture(0)  # 获得摄像头
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度

        cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            'camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.lastFrame = None

        with open('dictionary.json', 'r') as f:
            self.dictionary = json.load(f)

        print(f"cameraWidth: {self.width}, cameraHeight: {self.height}")
        checkDir(tmpFolder)

    def work(self):
        index = 0
        working = True
        while (self.cap.isOpened()):
            success, frame = self.cap.read()
            frame = frame[:, int((self.width-self.height)/2):int((self.width+self.height)/2)]

            # move = self.checkMove(frame)
            # 打印当前状态
            cv2.putText(
                frame, f"Identification = {index}", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow('camera', frame)

            cv2.imwrite(os.path.join(
                self.tmpFolder, "{:06d}.jpg".format(index)), frame)
            index += 1

            key = cv2.waitKey(1)
            if key == ord('w'):
                ansIndex = self.model.identification(self.tmpFolder)
                print(f"index: {ansIndex}, 含义:{self.dictionary[ansIndex]}")
                index = 0
                clearDir(self.tmpFolder)

        self.cap.release()
        clearDir(self.tmpFolder)

    def checkMove(self, frame: np.ndarray) -> bool:
        ret = False
        if (not self.lastFrame is None):
            ret: bool = (np.sum(frame-self.lastFrame) > 20000000)
            print(np.sum(frame-self.lastFrame))
        self.lastFrame = frame
        return ret

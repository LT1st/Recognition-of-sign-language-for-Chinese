import cv2
import os
import numpy as np
from Deduction import Deduction
import json
from queue import Queue


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

        # cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(
        #     'camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.lastFrame = None
        self.lastCode = 0
        self.imgQueue = Queue()

        with open('dictionary.json', 'r') as f:
            self.dictionary = json.load(f)

        print(f"cameraWidth: {self.width}, cameraHeight: {self.height}")
        checkDir(tmpFolder)

    def work(self):
        move = 0
        working = True
        while (self.cap.isOpened()):
            success, frame = self.cap.read()
            frame = frame[:, int((self.width-self.height)/2)                          :int((self.width+self.height)/2)]

            if (self.checkMove(frame)):
                move = 1
            move *= 0.9
            movement = (move > 0.15)

            # 打印当前状态
            cv2.putText(
                frame, f"movement = {movement}", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow('camera', frame)

            if (movement):
                self.imgQueue.put(frame)
            elif (not movement and self.imgQueue.qsize() > 40):
                index = 1
                while not self.imgQueue.empty():
                    cv2.imwrite(os.path.join(
                        self.tmpFolder, "{:06d}.jpg".format(index)), self.imgQueue.get())
                    index += 1

                ansIndex = self.model.identification(self.tmpFolder)
                print(f"index: {ansIndex}, 含义:{self.dictionary[ansIndex]}")
                index = 0
                clearDir(self.tmpFolder)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        clearDir(self.tmpFolder)

    def checkMove(self, frame: np.ndarray) -> bool:
        ret = False
        frame = cv2.resize(frame, (32, 32)).sum(axis=2)
        frame = frame * 0.01
        if (not self.lastFrame is None):
            nowCode = np.sum(frame.T * self.lastFrame)
            # 100是运动检测阈值，越小表示越容易捕捉到运动
            ret = (np.abs(nowCode-self.lastCode) > 50)
        self.lastFrame = frame
        self.lastCode = np.sum(frame.T * self.lastFrame)
        return ret

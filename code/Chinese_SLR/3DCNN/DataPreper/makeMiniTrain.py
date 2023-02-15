# 原有的train太大了，为了方便传输与训练，随机生成一个迷你版本的train
import os

trainPath = "./Train/"
miniTrainPath = "./miniTrain/"


def makeDirStruct():
    for i in range(0, 500):
        nowPath = os.path.join(miniTrainPath, "{:03d}".format(i))
        os.mkdir(nowPath)


def makeMiniTrain(size=20):
    for i in range(0, 500):
        nowPath = os.path.join(trainPath, "{:03d}".format(i))
        print(nowPath)

        copyFolder = os.listdir(nowPath)[:size]  # 随机抽取20个
        copyPath = [os.path.join(nowPath, x) for x in copyFolder]
        destPath = [os.path.join(
            miniTrainPath, "{:03d}".format(i), x) for x in copyFolder]

        for s, d in zip(copyPath, destPath):
            os.system(f"cp -r {s} {d}")


if __name__ == "__main__":
    os.mkdir(miniTrainPath)
    makeDirStruct()
    makeMiniTrain(80)

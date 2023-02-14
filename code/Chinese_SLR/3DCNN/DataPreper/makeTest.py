import os

trainPath = "./Train/"
testPath = "./Test/"

for i in range(0, 100):
    nowPath = os.path.join(trainPath, "{:03d}".format(i))
    print(nowPath)

    copyFolder = os.listdir(nowPath)[:10]
    copyPath = [os.path.join(nowPath, x) for x in copyFolder]
    destPath = [os.path.join(
        testPath, "{:03d}".format(i), x) for x in copyFolder]

    for s, d in zip(copyPath, destPath):
        os.replace(s, d)

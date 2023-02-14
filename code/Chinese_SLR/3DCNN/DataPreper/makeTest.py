import os

trainPath = "./Train/"
testPath = "./Test/"

for i in range(110, 500):
    nowPath = os.path.join(trainPath, str(i))
    print(nowPath)

    copyFolder = os.listdir(nowPath)[:10]
    copyPath = [os.path.join(nowPath, x) for x in copyFolder]
    destPath = [os.path.join(testPath, str(i), x) for x in copyFolder]

    for s, d in zip(copyPath, destPath):
        os.replace(s, d)

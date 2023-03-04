import os

from natsort import natsorted
input_path = r'F:\SLRdataset\all'
paths = []
names = []
label=[]
names_ = []
label_=[]
for root, dirs, fnames in natsorted(os.walk(input_path)):
    num = 0
    name_num = 0
    for dir in dirs:
        name_num+=1
        if name_num>50:
            break
        rootName = os.path.join(root, dir)
        for root_, dir_, _ in natsorted(os.walk(rootName)):
            sum=0
            for dir in dir_:
                sum+=1
                if sum == 10:
                    names_.append(dir)
                    label_.append(num)
                    continue
                if sum == 20:
                    names_.append(dir)
                    label_.append(num)
                    break
                names.append(dir)
                label.append(num)
            break
        num+=1

print(names)
print(label)
print(names_)
print(label_)
import csv
filePath = 'train_label.csv'

rows = zip(names,label)
with open(filePath, "w", newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

filePath_ = 'val_label.csv'

rows = zip(names_,label_)
with open(filePath_, "w", newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

import torch.nn as nn
import torchvision


def convert_relu_to_swish(model):       # 这个是批量更改激活函数的，原文中提到使用switsh()代替relu()
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SiLU(True))
        else:
            convert_relu_to_swish(child)


class r2plus1d_18(nn.Module):   # 3D CNN 模型，通过对标准库模型进行魔改得来
    def __init__(self, pretrained=True, num_classes=500, dropout_p=0.5):
        '''
            其中的num_classes参数是最终预测输出的种类数量
        '''
        super(r2plus1d_18, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes

        model = torchvision.models.video.r2plus1d_18(
            pretrained=self.pretrained)         # 从标准库取得模型

        # 魔改模型，去掉fc层
        modules = list(model.children())[:-1]
        self.r2plus1d_18 = nn.Sequential(*modules)  # 获得不含全链接层的网络
        convert_relu_to_swish(self.r2plus1d_18)  # 切换激活函数
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)
        self.dropout = nn.Dropout(dropout_p, inplace=True)

    def forward(self, x):
        out = self.r2plus1d_18(x)
        out = out.flatten(1)
        out = self.dropout(out)
        out = self.fc1(out)
        return out

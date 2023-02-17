from SignDataset import SignDataset
from Conv3D import r2plus1d_18
import torchvision.transforms as transforms
import torch


class Deduction:
    def __init__(self, modelPath: str) -> None:
        self.modelPath = modelPath
        # 数据
        transform = transforms.Compose([transforms.Resize([96, 96]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
        self.data = SignDataset(None, train=False, transform=transform)
        # 模型
        self.model = r2plus1d_18(pretrained=False)
        self.model.load_state_dict(torch.load(modelPath))  # 重新载入含有训练权重的模型
        self.model.eval()

    def identification(self, dataFolder: str = "./tmp/") -> int:
        input = torch.unsqueeze(self.data.getImages(dataFolder), dim=0)
        output = self.model(input)

        prediction: torch.Tensor = torch.max(output, dim=1).indices
        return prediction.item()

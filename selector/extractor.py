import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, base_model=models.resnet18, pretrained=True, alpha=0.5):
        super(FeatureExtractor, self).__init__()
        self.alpha = alpha
        self.original_branch = nn.Sequential(*list(base_model(pretrained=pretrained).children())[:-1])
        self.binary_branch = nn.Sequential(*list(base_model(pretrained=pretrained).children())[:-1])

    def forward(self, x1, x2):
        feat1 = self.original_branch(x1)
        feat1 = feat1.view(feat1.size(0), -1)

        feat2 = self.binary_branch(x2)
        feat2 = feat2.view(feat2.size(0), -1)

        # fused_features = torch.cat((feat1, feat2), dim=1)
        fused_features = self.alpha * feat1 + (1 - self.alpha) * feat2

        return fused_features


if __name__ == '__main__':
    import onnx
    import onnxruntime as ort
    import numpy as np
    import torch
    import random


    def set_seed(seed=42):
        # Python 的随机数种子
        random.seed(seed)
        # Numpy 的随机数种子
        np.random.seed(seed)
        # PyTorch 的随机数种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # 设置 GPU 随机种子
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
        # 确保每次结果的可复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # 设置随机种子
    set_seed(42)

    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    dummy_input = torch.tensor(dummy_input)
    model = FeatureExtractor(alpha=1.0)
    output = model(dummy_input, dummy_input)
    print(output)

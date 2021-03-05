import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


class WeirdEfficientNet(torch.nn.Module):

    def __init__(self, input_channels, output_dim, model_ver):#5 output classifications
        super().__init__()

        self.effnet = EfficientNet.from_pretrained(f'efficientnet-b{model_ver}')
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1536, output_dim)
        

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        
        x = self.effnet.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        output = self.out(self.dropout(x))

        return output

    def print_pretrained(self):
        modules = list(self.pretrained_layers.modules())
        for module in modules:
            print(module)
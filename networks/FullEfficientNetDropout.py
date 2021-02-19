import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


class FullEfficientNet(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_channels, output_dim, model_ver):#5 output classifications
        super().__init__()

        self.pretrained_layers = EfficientNet.from_pretrained(f'efficientnet-b{model_ver}', include_top=False)

        # self.pretrained_layers = nn.Sequential(*model_children)
        # for param in self.pretrained_layers.parameters():
        #     param.requires_grad = False

        self.fc_in_features = self.pretrained_layers._fc.in_features
        self.linear1 = nn.Linear(self.fc_in_features, 200)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(200, 100)
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(200, output_dim)
        

    def forward(self, x):
        x = self.pretrained_layers(x)
        x = torch.reshape(x, (-1, self.fc_in_features))
        x = F.relu(self.dropout1(self.linear1(x)))
        x = F.relu(self.dropout2(self.linear2(x)))
        x = self.linear3(x)
        return x

    def print_pretrained(self):
        modules = list(self.pretrained_layers.modules())
        for module in modules:
            print(module)
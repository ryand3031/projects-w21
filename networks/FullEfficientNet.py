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

        self.downsize = nn.Conv2d(3, 3, 2, stride=2)
        self.pool = nn.MaxPool2d(1)
        self.pretrained_layers = EfficientNet.from_pretrained(f'efficientnet-b{model_ver}', include_top=False)
        # model_children = list(pretrained_model.modules())
        # print(f'children length: {len(model_children)}')

        # self.pretrained_layers = nn.Sequential(*model_children)
        # for param in self.pretrained_layers.parameters():
        #     param.requires_grad = False

        # self.conv1 = nn.Conv2d(input_channels, 10, 5, padding=2)
        # self.conv2 = nn.Conv2d(10, 30, 7, padding=3)
        # self.maxPool = nn.MaxPool2d(4)
        # self.conv3 = nn.Conv2d(30, 5, 7, padding=3)
        # self.linear1 = nn.Linear(14 * 14 * 5, 200)  #980 to 200
        # self.linear2 = nn.Linear(200, 100)
        # self.linear3 = nn.Linear(100, 25)

        self.fc_in_features = self.pretrained_layers._fc.in_features
        self.linear1 = nn.Linear(self.fc_in_features, 64)
        self.linear2 = nn.Linear(64, output_dim)
        

    def forward(self, x):
        x = self.pool(F.relu(self.downsize(x)))
        x = self.pretrained_layers(x)
        x = torch.reshape(x, (-1, self.fc_in_features))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def print_pretrained(self):
        modules = list(self.pretrained_layers.modules())
        for module in modules:
            print(module)
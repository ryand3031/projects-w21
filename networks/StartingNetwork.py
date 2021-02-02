import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_channels, output_dim):#5 output classifications
        super().__init__()

        self.pretrained_layers = EfficientNet.from_pretrained('efficientnet-b0', include_top=False)
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

        self.linear1 = nn.Linear(1280, 64)
        self.linear2 = nn.Linear(64, output_dim)
        

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x)) 
        # x = self.maxPool(x)
        # x = F.relu(self.conv3(x)) 
        # x = self.maxPool(x)
        # x = x.reshape([-1, 14 * 14 * 5])
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))

        x = self.pretrained_layers(x)
        x = torch.reshape(x, (-1, 1280))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def print_pretrained(self):
        modules = list(self.pretrained_layers.modules())
        for module in modules:
            print(module)

if __name__ == '__main__':
    model = StartingNetwork(3, 1000)
    test_input = torch.ones(1, 3, 224, 224)
    model(test_input)
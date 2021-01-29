import torch
import torch.nn as nn
import torch.nn.functional as F



class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_channels, output_dim):#5 output classifications
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 30, 7, padding=3)
        self.maxPool = nn.MaxPool2d(4)
        self.conv3 = nn.Conv2d(30, 5, 7, padding=3)
        self.linear1 = nn.Linear(14 * 14 * 5, 200)  #980 to 200
        self.linear2 = nn.Linear(200, 100)
        self.linear3 = nn.Linear(100, 25)
        self.linear4 = nn.Linear(25, output_dim)
            

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = self.maxPool(x)
        x = F.relu(self.conv3(x)) 
        x = self.maxPool(x)
        x = x.reshape([-1, 14 * 14 * 5])
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
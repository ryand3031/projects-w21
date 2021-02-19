import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class FullResNextModelDropout(torch.nn.Module):
    """
    Based off of resnext
    """

    def __init__(self, output_dim):#5 output classifications
        super().__init__()

        self.pretrained_layers = torchvision.models.resnext101_32x8d(pretrained=True, progress=True)

        # for param in self.pretrained_layers.parameters():
        #     param.requires_grad = False

        self.linear1 = nn.Linear(self.pretrained_layers.fc.in_features, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, output_dim)
        self.fc = nn.Sequential(self.linear1, nn.Dropout(), nn.ReLU(), self.linear2, nn.Dropout(), nn.ReLU(), self.linear3)
        self.pretrained_layers.fc = self.fc
        

    def forward(self, x):
        return self.pretrained_layers(x)

    def print_pretrained(self):
        modules = list(self.pretrained_layers.modules())
        for module in modules:
            print(module)

if __name__ == '__main__':
    model = FullResNextModelDropout(5)
    test_input = torch.ones(1, 3, 224, 224)
    model(test_input)
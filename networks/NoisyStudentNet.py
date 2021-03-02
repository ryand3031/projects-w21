import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


class NoisyStudentNet(torch.nn.Module):

    def __init__(self, output_dim, model_ver, dropout=False, fine_tune=True):#5 output classifications
        super().__init__()
        
        self.pretrained_layers = torch.hub.load('rwightman/gen-efficientnet-pytorch', f'tf_efficientnet_b{model_ver}_ns', pretrained=True, force_reload=True)

        if not fine_tune:
            for param in self.pretrained_layers.parameters():
                param.requires_grad = False

        self.fc_in_features = self.pretrained_layers.classifier.in_features

        self.linear1 = nn.Linear(self.fc_in_features, 200)
        self.linear2 = nn.Linear(200, 100)
        self.linear3 = nn.Linear(100, output_dim)
        
        if dropout:
            self.fc = nn.Sequential(self.linear1, nn.Dropout(), nn.ReLU(), self.linear2, nn.Dropout(), nn.ReLU(), self.linear3)
        else:
            self.fc = nn.Sequential(self.linear1, nn.ReLU(), self.linear2, nn.ReLU(), self.linear3)
        
        self.pretrained_layers.classifier = self.fc
        

    def forward(self, x):
        x = self.pretrained_layers(x)
        # x = torch.reshape(x, (-1, self.fc_in_features))
        # x = F.relu(self.dropout1(self.linear1(x)))
        # x = F.relu(self.dropout2(self.linear2(x)))
        # x = self.linear3(x)
        return x

    def print_pretrained(self):
        modules = list(self.pretrained_layers.modules())
        for module in modules:
            print(module)
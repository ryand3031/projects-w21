import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class EssembleWrapper(torch.nn.Module):
    def __init__(self, models, sizes, device):
        super().__init__()
        self.models = models
        self.resizes = sizes
        self.device = device
        for i in range(len(self.resizes)):
            self.resizes[i] = transforms.Resize(sizes[i])

    def forward(self, x):
        sum = torch.zeros(x.shape[0], 5).to(self.device)
        for model, resize in zip(self.models, self.resizes):
            x = resize(x)
            sum += model(x)
        return sum


if __name__ == "__main__":
    from FullResNextModel import FullResNextModel
    from WeirdEfficientNet import WeirdEfficientNet
    from ..train_functions.starting_train import evaluate
    from ..datasets.AugmentedDataset import AugmentedDataset
    
    resnext = FullResNextModel(5)
    efficientnet = WeirdEfficientNet(3, 5, 3)

    resnext.load_state_dict(torch.load('../trained_resnext.pt'))
    efficientnet.load_state_dict(torch.load('../trained_efficientnet.pt'))

    ensemble = EssembleWrapper([resnext, efficientnet], [224, 300])

    val_dataset = AugmentedDataset(train=False, resize=350, centerCrop=300)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

    print(evaluate(val_loader, model, nn.CrossEntropyLoss(), device, use_tta=False))

    # data = torch.rand(10, 3, 300, 300)
    # predictions = ensemble(data)
    # print(predictions.argmax(axis = 1))
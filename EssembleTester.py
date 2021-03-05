import torch
import torch.nn as nn
from networks.FullResNextModel import FullResNextModel
from networks.FullEfficientNet import FullEfficientNet
from networks.WeirdEfficientNet import WeirdEfficientNet
from train_functions.starting_train import evaluate
from datasets.AugmentedDataset import AugmentedDataset
from networks.EssembleWrapper import EssembleWrapper

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

resnext = FullResNextModel(5).to(device)
efficientnetb3 = WeirdEfficientNet(3, 5, 3).to(device)
efficientnetb4 = FullEfficientNet(3, 5, 4).to(device)

resnext.load_state_dict(torch.load('./trained_models/trained_resnext.pt'))
efficientnetb3.load_state_dict(torch.load('./trained_models/trained_efficientnetb3.pt'))
efficientnetb4.load_state_dict(torch.load('./trained_models/trained_efficientnetb4.pt'))

essemble = EssembleWrapper([resnext, efficientnetb3, efficientnetb4], [224, 300, 224], device).to(device)

val_dataset = AugmentedDataset(train=False, resize=350, centerCrop=300)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

print(evaluate(val_loader, essemble, nn.CrossEntropyLoss(), device, use_tta=False))

# data = torch.rand(10, 3, 300, 300)
# predictions = essemble(data)
# print(predictions.argmax(axis = 1))
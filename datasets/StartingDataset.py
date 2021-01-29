import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 3x224x224 images.
    """

    def __init__(self):
        self.transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # pandas dataframe that stores image names and labels
        self.df = pd.read_csv('./data/train.csv').sample(frac=1)

    def __getitem__(self, index):
        img = self.df.iloc[index]
        inputs = self.transformations(Image.open(f"./data/train_images/{img['image_id']}"))
        label = img['label']

        return inputs, label

    def __len__(self):
        return len(self.df)
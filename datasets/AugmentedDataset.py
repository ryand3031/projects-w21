import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 3x224x224 images.
    """

    def __init__(self, train=False, seed=1):
        self.transformations = transforms.Compose([
            transforms.Resize(510),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.random_transformations = transforms.Compose([
            transforms.RandomRotation(90, expand=False, fill=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
            torch.nn.Dropout2d(0.1)
        ])

        self.train = train


        # pandas dataframe that stores image names and labels
        self.df = pd.read_csv('./data/train.csv').sample(frac=1, random_state=seed)

        test_section = int(0.8 * len(self.df))
        if train:
            self.df = self.df.iloc[:test_section]
            self.df_transformed = self.df.copy()
            self.df = pd.concat([self.df, self.df_transformed])
        else:
            self.df = self.df.iloc[test_section:]

    def __getitem__(self, index):
        img = self.df.iloc[index]

        inputs = Image.open(f"./data/train_images/{img['image_id']}")
        inputs = self.transformations(inputs)
        if self.train and index >= self.__len__() / 2:
            inputs = self.random_transformations(inputs)

        return inputs, img['label']

    def __len__(self):
        return len(self.df)
        
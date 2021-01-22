import torch


# random comment
class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        pass

    def __getitem__(self, index):
        inputs = torch.zeros([3, 224, 224])
        label = 0

        return inputs, label

    def __len__(self):
        return 10000

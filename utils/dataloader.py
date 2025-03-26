import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataLoader:
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert to FloatTensor of shape (C*H*W) in range [0.0, 1.0]
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1.0, 1.0]
            ]
        )

    def get_dataloader(self, train=True):
        dataset = datasets.MNIST(
            root="./data", train=train, download=True, transform=self.transform
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        return dataloader

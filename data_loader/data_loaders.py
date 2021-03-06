from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(), # img to tensor
            transforms.Normalize((0.1307,), (0.3081,)) # mean = 1307, std = 3081 정규화
        ])
        self.data_dir = data_dir # 경로
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm) # MNIST dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers) # BaseDataLoader

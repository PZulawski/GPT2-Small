import torch
from torch.utils.data import DataLoader

class PrefetchDataLoader():
    def __init__(self, dataloader: DataLoader, device):
        batch_shape = next(iter(dataloader))[0].shape
        data = list(dataloader)
        self.data = torch.stack([batch[0] for batch in data if batch[0].shape == batch_shape]).to(device)
        self.target = torch.stack([batch[1] for batch in data if batch[1].shape == batch_shape]).to(device)
        self.ordering = torch.randperm(len(self.data))
        self.counter = 0

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    def __next__(self):
        if self.counter == len(self.data):
            self.ordering = torch.randperm(len(self.data))
            self.counter = 0
        self.counter += 1
        return self.__getitem__(self.ordering[self.counter - 1])
    
    def __len__(self):
        return len(self.data)
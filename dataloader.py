import tiktoken
import torch
from torch.utils.data import DataLoader

class DataLoader(DataLoader):
    def __init__(self):
        
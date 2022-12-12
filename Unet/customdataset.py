import torch
import numpy as np
from torch.utils import data
from tqdm.notebook import tqdm

class SegmentationDataSet1(data.Dataset):
    """Most basic image segmentation dataset."""

    def __init__(self, inputs: np.array, targets: np.array, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        
        x = self.inputs[index]
        y = self.targets[index]
        
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(self.inputs, self.targets)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y
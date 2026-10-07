import os
import pickle
import warnings
from typing import Optional, Callable
from torch_geometric.data import Dataset, HeteroData

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DroneDataset(Dataset):
    def __init__(self,
                 root: str,
                 dataset: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 small_data: bool = False) -> None:
        super().__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)
        assert split in ['train', 'val', 'test'], 'Split must be one of [train, val, test]'

        self.root = root
        self.dataset = dataset
        self.split = split
        self.path = os.path.join(self.root, self.dataset, self.split)
        self.files = os.listdir(self.path)
        self.files = sorted(self.files)  # sort files for consistency across operating systems

        if small_data:
            self.files = self.files[:100]

        self._num_samples = len(self.files)

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        file_path = os.path.join(self.path, self.files[idx])
        with open(file_path, 'rb') as f:
            return HeteroData(pickle.load(f))

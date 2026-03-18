import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class FullVideoDataset(Dataset):

    def __init__(self, list_path, label_map):
        self.data = pd.read_csv(list_path)
        self.label_map = label_map
        self.paths = self.data['path']
        self.labels = self.data['label']

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        try:
            feature = np.load(path)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            return None, None, None
        feature_tensor = torch.from_numpy(feature).float()
        return feature_tensor, label, feature_tensor.shape[0]


def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.empty(0), [], torch.empty(0)
    return batch[0]
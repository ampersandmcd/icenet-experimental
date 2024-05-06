import sys
sys.path.insert(0, '../icenet')  # import local IceNet rather than pip IceNet
from icenet.data.dataset import IceNetDataSet
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class IceNetDataSetPyTorch(Dataset):
    def __init__(self,
                 configuration_path: str,
                 mode: str,
                 batch_size: int = 1,
                 shuffling: bool = False,
                ):
        self._ds = IceNetDataSet(configuration_path=configuration_path,
                                 batch_size=batch_size,
                                 shuffling=shuffling)
        self._dl = self._ds.get_data_loader()

        # check mode option
        if mode not in ["train", "val", "test"]:
            raise ValueError("mode must be either 'train', 'val', 'test'")
        self._mode = mode

        self._dates = self._dl._config["sources"]["osisaf"]["dates"][self._mode]

    def __len__(self):
        return self._ds._counts[self._mode]
    
    def __getitem__(self, idx):
        x, y, sw = self._dl.generate_sample(date=pd.Timestamp(self._dates[idx].replace('_', '-')),
                                        parallel=False)
        x, y, sw = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(sw)
        x = x.permute(2, 0, 1)  # put channel to first dimension for pytorch
        y = y.squeeze(-1).permute(2, 0, 1)  # collapse repeated dimension and put channel first
        sw = sw.squeeze(-1).permute(2, 0, 1)  # collapse repeated dimension and put channel first
        return x, y, sw

    def get_data_loader(self):
        return self._ds.get_data_loader()
    
    @property
    def dates(self):
        return self._dates

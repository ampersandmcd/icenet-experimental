import sys
sys.path.insert(0, '../icenet')  # import local IceNet rather than pip IceNet
from icenet.data.dataset import IceNetDataSet
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torchmetrics import Metric


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


class BinaryAccuracy(Metric):
    """
    Binary accuracy metric for use at multiple leadtimes.
    """    

    # Set class properties
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list):
        """
        Construct a binary accuracy metric for use at multiple leadtimes.
        :param leadtimes_to_evaluate: A list of leadtimes to consider
            e.g., [0, 1, 2, 3, 4, 5] to consider all six months in accuracy computation or
            e.g., [0] to only look at the first month's accuracy
            e.g., [5] to only look at the sixth month's accuracy
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("possible", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        # anything greater than 0.15 concentration is ice
        preds = (preds > 0.15).long()
        target = (target > 0.15).long()
        self.correct += torch.sum(preds[:, :, :, self.leadtimes_to_evaluate] == target[:, :, :, self.leadtimes_to_evaluate])
        self.possible += torch.sum(torch.where(sample_weight[:, :, :, self.leadtimes_to_evaluate] > 0, 1, 0))  # binary mask

    def compute(self):
        return self.correct.float() / self.possible


class SIEError(Metric):
    """
    Sea Ice Extent error metric (in km^2) for use at multiple leadtimes.
    """ 

    # Set class properties
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list):
        """
        Construct an SIE error metric (in km^2) for use at multiple leadtimes.
        :param leadtimes_to_evaluate: A list of leadtimes to consider
            e.g., [0, 1, 2, 3, 4, 5] to consider all six months in computation or
            e.g., [0] to only look at the first month
            e.g., [5] to only look at the sixth month
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("pred_sie", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("true_sie", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        # anything greater than 0.15 concentration is ice
        self.pred_sie += torch.sum(torch.where(preds[:, :, :, self.leadtimes_to_evaluate] > 0.15, 1, 0))
        self.true_sie += torch.sum(torch.where(target[:, :, :, self.leadtimes_to_evaluate] > 0.15, 1, 0))

    def compute(self):
        return (self.pred_sie - self.true_sie) * 25**2 # each pixel is 25x25 km
    

class SIAError(Metric):
    """
    Sea Ice Area error metric (in km^2) for use at multiple leadtimes.
    """ 

    # Set class properties
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list):
        """
        Construct an SIA error metric (in km^2) for use at multiple leadtimes.
        :param leadtimes_to_evaluate: A list of leadtimes to consider
            e.g., [0, 1, 2, 3, 4, 5] to consider all six months in computation or
            e.g., [0] to only look at the first month
            e.g., [5] to only look at the sixth month
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("pred_sia", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("true_sia", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        # sea ice area is integral of concentration
        self.pred_sia += torch.sum(preds[:, :, :, self.leadtimes_to_evaluate])
        self.true_sia += torch.sum(target[:, :, :, self.leadtimes_to_evaluate])

    def compute(self):
        return (self.pred_sia - self.true_sia) * 25**2 # each pixel is 25x25 km


class RMSError(Metric):
    """
    RMSE for use at multiple leadtimes.
    """

    # Set class properties
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list):
        """
        Construct an RMSE error metric (in km^2) for use at multiple leadtimes.
        :param leadtimes_to_evaluate: A list of leadtimes to consider
            e.g., [0, 1, 2, 3, 4, 5] to consider all six months in computation or
            e.g., [0] to only look at the first month
            e.g., [5] to only look at the sixth month
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("diffs", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pixels", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        self.diffs += torch.sum((preds[:, :, :, self.leadtimes_to_evaluate] - target[:, :, :, self.leadtimes_to_evaluate])**2)
        self.pixels += torch.sum(torch.where(sample_weight[:, :, :, self.leadtimes_to_evaluate] > 0, 1, 0))

    def compute(self):
        return torch.sqrt(self.diffs / self.pixels)


class MAError(Metric):
    """
    MAE for use at multiple leadtimes.
    """

    # Set class properties
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list):
        """
        Construct an MAE error metric (in km^2) for use at multiple leadtimes.
        :param leadtimes_to_evaluate: A list of leadtimes to consider
            e.g., [0, 1, 2, 3, 4, 5] to consider all six months in computation or
            e.g., [0] to only look at the first month
            e.g., [5] to only look at the sixth month
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("diffs", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pixels", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        self.diffs += torch.sum(torch.abs(preds[:, :, :, self.leadtimes_to_evaluate] - target[:, :, :, self.leadtimes_to_evaluate]))
        self.pixels += torch.sum(torch.where(sample_weight[:, :, :, self.leadtimes_to_evaluate] > 0, 1, 0))

    def compute(self):
        return self.diffs / self.pixels

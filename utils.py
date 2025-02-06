import sys
sys.path.insert(0, '../icenet')  # import local IceNet rather than pip IceNet
from icenet.data.dataset import IceNetDataSet
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import torch
from torchmetrics import Metric
import logging
from torchdata.datapipes.iter import FileOpener
import numpy as np
import scipy.stats as stats


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
        self._i = 0

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
        return x, y, sw, self._dates[idx].replace('_', '-')

    def get_data_loader(self):
        return self._ds.get_data_loader()
    
    @property
    def dates(self):
        return self._dates
    

class IterableIceNetDataSetPyTorch(IterableDataset):
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
        self._i = 0

        # check mode option
        if mode not in ["train", "val", "test"]:
            raise ValueError("mode must be either 'train', 'val', 'test'")
        self._mode = mode

        self._dates = self._dl._config["sources"]["osisaf"]["dates"][self._mode]

        # load from tfrecords if processed
        self._processed = False
        if (len(self._ds.train_fns) + len(self._ds.val_fns) + len(self._ds.test_fns)) > 0:
            self._processed = True
            mode_to_filenames = {"train": self._ds.train_fns, "val": self._ds.val_fns, "test": self._ds.test_fns}
            filenames = mode_to_filenames[self._mode]
            self._datapipe = FileOpener(filenames, mode="b", length=len(filenames))

    def __len__(self):
        return self._ds._counts[self._mode]

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._i < len(self):
            self._i += 1
            if self._processed:
                for sample in self._datapipe.load_from_tfrecord():
                    x, y, sw = sample["x"].numpy(), sample["y"].numpy(), sample["sample_weights"].numpy()
                    x = x.reshape(self._ds.shape + (-1,))
                    y = y.reshape(self._ds.shape + (-1,))
                    sw = sw.reshape(self._ds.shape + (-1,))
                    x, y, sw = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(sw)
                    x = x.permute(2, 0, 1)  # put channel to first dimension for pytorch
                    y = y.squeeze(-1).permute(2, 0, 1)  # collapse repeated dimension and put channel first
                    sw = sw.squeeze(-1).permute(2, 0, 1)  # collapse repeated dimension and put channel first
                    return x, y, sw
            else:
                x, y, sw = self._dl.generate_sample(date=pd.Timestamp(self._dates[self._i].replace('_', '-')),
                                            parallel=False)
                x, y, sw = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(sw)
                x = x.permute(2, 0, 1)  # put channel to first dimension for pytorch
                y = y.squeeze(-1).permute(2, 0, 1)  # collapse repeated dimension and put channel first
                sw = sw.squeeze(-1).permute(2, 0, 1)  # collapse repeated dimension and put channel first
                return x, y, sw
        else:
           raise StopIteration()

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
        self.add_state("pixels", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, t, h, w)
        # anything greater than 0.15 concentration is ice
        preds = preds[:, self.leadtimes_to_evaluate, :, :]
        target = target[:, self.leadtimes_to_evaluate, :, :]
        mask = torch.where(sample_weight[:, self.leadtimes_to_evaluate, :, :] > 0, 1, 0)
        bin_preds = torch.where(preds > 0.15, 1, 0)
        bin_target = torch.where(target > 0.15, 1, 0)
        self.correct += torch.sum(torch.logical_and(bin_preds == bin_target, mask))
        self.pixels += torch.sum(mask)  # binary mask

    def compute(self):
        return self.correct / self.pixels


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
        self.add_state("diffs", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("forecasts", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, t, h, w)
        # anything greater than 0.15 concentration is ice
        preds = preds[:, self.leadtimes_to_evaluate, :, :]
        target = target[:, self.leadtimes_to_evaluate, :, :]
        mask = torch.where(sample_weight[:, self.leadtimes_to_evaluate, :, :] > 0, 1, 0)
        bin_preds = torch.where(preds > 0.15, 1, 0)
        bin_target = torch.where(target > 0.15, 1, 0)
        # reduce with sign over single field, reduce abs over timesteps and batches
        diffs_by_field = torch.sum((bin_preds - bin_target) * mask, dim=(2, 3))
        self.diffs += torch.sum(torch.abs(diffs_by_field))
        b, t, h, w = target.shape
        self.forecasts += b * t

    def compute(self):
        return self.diffs / self.forecasts * 25**2  # each pixel is 25x25 km
    

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
        self.add_state("diffs", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("forecasts", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, t, h, w)
        # sea ice area is integral of concentration
        preds = preds[:, self.leadtimes_to_evaluate, :, :]
        target = target[:, self.leadtimes_to_evaluate, :, :]
        mask = torch.where(sample_weight[:, self.leadtimes_to_evaluate, :, :] > 0, 1, 0)
        # reduce with sign over single field, reduce abs over timesteps and batches
        diffs_by_field = torch.sum((preds - target) * mask, dim=(2, 3))
        self.diffs += torch.sum(torch.abs(diffs_by_field))
        b, t, h, w = target.shape
        self.forecasts += b * t

    def compute(self):
        return self.diffs / self.forecasts * 25**2  # each pixel is 25x25 km


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
        preds = preds[:, self.leadtimes_to_evaluate, :, :]
        target = target[:, self.leadtimes_to_evaluate, :, :]
        mask = torch.where(sample_weight[:, self.leadtimes_to_evaluate, :, :] > 0, 1, 0)
        self.diffs += torch.sum((preds - target)**2)
        self.pixels += torch.sum(mask)

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
        # preds and target are shape (b, t, h, w)
        preds = preds[:, self.leadtimes_to_evaluate, :, :]
        target = target[:, self.leadtimes_to_evaluate, :, :]
        mask = torch.where(sample_weight[:, self.leadtimes_to_evaluate, :, :] > 0, 1, 0)
        self.diffs += torch.sum(torch.abs(preds - target))
        self.pixels += torch.sum(mask)
        
    def compute(self):
        return self.diffs / self.pixels


def compute_spectrum2d(data):
    # from https://github.com/robbiewatt1/ClimateDiffuse/blob/main/inference/compute_spectrum.py
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    N, N_y, N_x = data.shape
    if N_x == 2 * N_y:
        data1 = data[:, :, :N_y]
        data2 = data[:, :, N_y:]
        data = np.concatenate((data1, data2), axis=0)
    N, N_y, N_x = data.shape

    # Take FFT and take amplitude
    fourier_image = np.fft.fftn(data, axes=(1,2))
    fourier_amplitudes = np.abs(fourier_image)**2

    # Get kx and ky
    kfreq_x = np.fft.fftfreq(N_x) * N_x
    kfreq_y = np.fft.fftfreq(N_y) * N_y

    # Combine into one wavenumber for both directions
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = np.repeat(knrm[np.newaxis, ...], repeats=N, axis=0)

    # Flatten arrays
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    # Get k-bins and mean amplitude within each bin
    kbins = np.arange(0.5, N_x//2, 1)
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)

    # Multiply by volume of bin
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    # Get center of k-bin for plotting
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    return (kvals, Abins)


def compute_spectrum2d_alternative(data):

    npix = data.shape[0]

    fourier_image = np.fft.fftn(data)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, Abins
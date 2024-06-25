import os
import pandas as pd
import numpy as np
import torch
from utils import IterableIceNetDataSetPyTorch

if "datasets.ipynb" in os.listdir():
    os.chdir("../")
print("Running in {}".format(os.getcwd()))

implementation = "dask"
dataset_config = "dataset_config.exp23_south.json"
lag = 1

ds = IterableIceNetDataSetPyTorch(dataset_config, "test", batch_size=4, shuffling=False)
dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

for i, batch in enumerate(dl):
    x, y, sw = batch
    print(x, y, sw)
    print(i)

import os
import pandas as pd
import numpy as np
from icenet.data.processors.era5 import IceNetERA5PreProcessor
from icenet.data.processors.meta import IceNetMetaPreProcessor
from icenet.data.processors.osi import IceNetOSIPreProcessor
from icenet.data.loaders import IceNetDataLoaderFactory
from utils import IterableIceNetDataSetPyTorch

if "datasets.ipynb" in os.listdir():
    os.chdir("../")
print("Running in {}".format(os.getcwd()))

implementation = "dask"
dataset_config = "dataset_config.exp23_south.json"
lag = 1

ds = IterableIceNetDataSetPyTorch(dataset_config, "val", batch_size=4, shuffling=False)

# x, y, sw = next(iter(ds))
# print(x)

for i, batch in enumerate(ds):
    x, y, sw = batch
    print(x, y, sw)
    print(i)

import os
import pandas as pd
import numpy as np
from icenet.data.processors.era5 import IceNetERA5PreProcessor
from icenet.data.processors.meta import IceNetMetaPreProcessor
from icenet.data.processors.osi import IceNetOSIPreProcessor
from icenet.data.loaders import IceNetDataLoaderFactory

if "datasets.ipynb" in os.listdir():
    os.chdir("../")
print("Running in {}".format(os.getcwd()))

processing_dates = dict(
    train=[pd.to_datetime(el) for el in pd.date_range("2020-01-01", "2020-01-01")],
    val=[pd.to_datetime(el) for el in pd.date_range("2020-01-01", "2020-01-01")],
    test=[pd.to_datetime(el) for el in pd.date_range("2020-01-01", "2020-01-01")],
)
processed_name = "single_day"

pp = IceNetERA5PreProcessor(
    ["uas", "vas"],
    ["tas", "zg500", "zg250"],
    processed_name,
    processing_dates["train"],
    processing_dates["val"],
    processing_dates["test"],
    linear_trends=tuple(),
    north=False,
    south=True
)
osi = IceNetOSIPreProcessor(
    ["siconca"],
    [],
    processed_name,
    processing_dates["train"],
    processing_dates["val"],
    processing_dates["test"],
    linear_trends=tuple(),
    north=False,
    south=True
)
meta = IceNetMetaPreProcessor(
    processed_name,
    north=False,
    south=True
)
pp.init_source_data(
    lag_days=1,
)
pp.process()
osi.init_source_data(
    lag_days=1,
)
osi.process()
meta.process()

implementation = "dask"
loader_config = "loader.single_day.json"
dataset_name = "single_day"
lag = 1

dl = IceNetDataLoaderFactory().create_data_loader(
    implementation,
    loader_config,
    dataset_name,
    lag,
    n_forecast_days=1,
    north=False,
    south=True,
    output_batch_size=1,
    generate_workers=1
)

x, y, sw = dl.generate_sample(pd.Timestamp("2020-12-01"))
diff = np.sum((x[:, :, 2] - y.squeeze())**2)
print(diff)  # should not be zero

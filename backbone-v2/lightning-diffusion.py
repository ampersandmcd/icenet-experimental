import os
import sys
import pandas as pd
import lightning.pytorch as pl
import wandb
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.ops.focal_loss import sigmoid_focal_loss
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from utils import BinaryAccuracy, SIEError, SIAError, RMSError, MAError
from utils import IceNetDataSetPyTorch, IterableIceNetDataSetPyTorch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# trade off speed and performance depending on gpu
torch.set_float32_matmul_precision("medium")
# torch.set_float32_matmul_precision("high")

### PyTorch Lightning modules:
# --------------------------------------------------------------------

class LitDiffusion(pl.LightningModule):
    """
    A LightningModule wrapping the diffusion implementation of IceNet.
    :param resolution: size of the forecast field
    :param in_channels: number of channels coming into the network = predictors + noise
    :param out_channels: number of channels coming out of the network = forecast steps at once
    :param block_out_channels: channels per UNet block, rough control over network size
    :param forecast_residuals: if True, forecasts ice directly, if False, forecasts residuals between steps
    :param noise_scheduler: HF diffusers noise scheduler object
    """
    def __init__(self,
                 resolution=432,
                 in_channels=2,
                 out_channels=1,
                 block_out_channels=(32, 32, 64),
                 forecast_residuals=False,
                 num_train_timesteps=1000,
                 num_inference_timesteps=50,
                 num_samples=3,
                 noise_scheduler=DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"),
                 learning_rate=1e-3):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.forecast_residuals = forecast_residuals
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.num_samples = num_samples
        self.noise_scheduler = noise_scheduler
        self.learning_rate = learning_rate
        self.model = UNet2DModel(
            sample_size=resolution,  # the target image resolution
            in_channels=in_channels,  # target channels + input channels
            out_channels=out_channels,  # target channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",  
                "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
            ),
            up_block_types=(
                "UpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )
        val_metrics = {
            "val_rmse": RMSError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_mae": MAError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_accuracy": BinaryAccuracy(leadtimes_to_evaluate=list(range(out_channels))),
            "val_sieerror": SIEError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_siaerror": SIAError(leadtimes_to_evaluate=list(range(out_channels)))
        }
        self.val_metrics = MetricCollection(val_metrics)
        test_metrics = {
            "val_rmse": RMSError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_mae": MAError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_accuracy": BinaryAccuracy(leadtimes_to_evaluate=list(range(out_channels))),
            "val_sieerror": SIEError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_siaerror": SIAError(leadtimes_to_evaluate=list(range(out_channels)))
        }
        self.test_metrics = MetricCollection(test_metrics)

        self.save_hyperparameters()

    def forward(self, sample_y, t, condition_x):
        """
        Implement forward function.
        :param sample_y: Partially noised target.
        :param t: Timestep in denoising process.
        :param condition_x: Conditioning information.
        :return: Outputs of model.
        """
        net_input = torch.cat((sample_y, condition_x), 1)  # (bs, target_c + input_c, resolution, resolution)
        return self.model(net_input, t).sample  # (bs, target_c, resolution, resolution)

    def training_step(self, batch, batch_idx: int):
        """
        Perform a pass through a batch of training data.
        :param batch: Batch of input, output, weight triplets.
        :param batch_idx: Index of batch.
        :return: Loss from this batch of data for use in backprop.
        """
        x, y, sample_weight = batch
        if self.forecast_residuals:
            y = y - x
        noise = torch.randn_like(y)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x.shape[0],)).long().to(self.device)
        noisy_y = self.noise_scheduler.add_noise(y, noise, timesteps)

        pred = self.forward(noisy_y, timesteps, x)
        loss = F.mse_loss(pred, noise)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        if batch_idx > 0:  # only validate first batch since validation is so slow
            return

        x, y, sample_weight = batch
        if self.forecast_residuals:
            y = y - x

        # code for multi-batch sampling
        # samples = []
        # for i in range(self.num_samples):  # for each condition in batch, generate multiple samples

        #     noisy_y = torch.randn(x.shape[0], self.out_channels,
        #                           self.resolution, self.resolution).to(self.device)

        #     for t in tqdm(self.noise_scheduler.timesteps):
        #         with torch.no_grad():
        #             residual = self.forward(noisy_y, t, x)
        #         noisy_y = self.noise_scheduler.step(residual, t, noisy_y).prev_sample
        #     samples.append(noisy_y)
        # samples = torch.stack(samples)  # shape (num_samples, batch_size, timesteps, y, x)

        # code for single-batch sampling
        # stack num_samples together with batches for faster processing
        bs = x.shape[0]
        noisy_y = torch.randn(self.num_samples * bs, self.out_channels,
                              self.resolution, self.resolution).to(self.device)
        x = torch.vstack([x]*self.num_samples)

        for t in tqdm(self.noise_scheduler.timesteps):
            with torch.no_grad():
                residual = self.forward(noisy_y, t, x)
            noisy_y = self.noise_scheduler.step(residual, t, noisy_y).prev_sample

        # unstack num_samples from batches and take mean/std
        samples = noisy_y.reshape(self.num_samples, bs, self.out_channels, self.resolution, self.resolution)
        means = samples.mean(dim=0)  # take ensemble mean across num_samples dimension
        stds = samples.std(dim=0)  # take ensemble std across num_samples dimension

        # messy plotting code
        nrows, ncols = bs, self.num_samples + 3
        fig, ax = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        for i in range(nrows):
            for j in range(self.num_samples):
                ax[i][j].imshow(samples[j][i].squeeze().cpu())
                ax[i][j].set_title(f"Condition {i} Sample {j}")
            ax[i][self.num_samples].imshow(means[i].squeeze().cpu())
            ax[i][self.num_samples].set_title(f"Mean for Condition {i}")
            ax[i][self.num_samples + 1].imshow(stds[i].squeeze().cpu())
            ax[i][self.num_samples + 1].set_title(f"Std for Condition {i}")
            ax[i][self.num_samples + 2].imshow(y[i].squeeze().cpu())
            ax[i][self.num_samples + 2].set_title(f"Truth for Condition {i}")
        plt.tight_layout()
        self.logger.log({"Sample": plt})
                    
        self.val_metrics.update(noisy_y, y)
    
    def on_validation_epoch_start(self):
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.val_metrics.reset()
        self.noise_scheduler.set_timesteps(self.num_train_timesteps)
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        torch.save(self.model, f"results/lightning-diffusion-epoch-{self.current_epoch}-{dt}.pth")

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.test_metrics.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return [opt], []  # add schedulers to second list if desired


### Benchmark models:
# --------------------------------------------------------------------


def linear_trend_forecast(forecast_month, n_linear_years='all', da=None, dataset='obs'):
    '''
    Returns a simple sea ice forecast based on a gridcell-wise linear extrapolation.

    Parameters:
    forecast_month (datetime.datetime): The month to forecast

    n_linear_years (int or str): Number of past years to use for linear trend
    extrapolation.

    da (xr.DataArray): xarray data array to use instead of observational
    data (used for setting up CMIP6 pre-training linear trend inputs in IceUNetDataPreProcessor).

    dataset (str): 'obs' or 'cmip6'. If 'obs', missing observational SIC months
    will be skipped

    Returns:
    output_map (np.ndarray): The output SIC map predicted
    by fitting a least squares linear trend to the past n_linear_years
    for the month being predicted.

    sie (np.float): The predicted sea ice extend (SIE).
    '''

    if da is None:
        with xr.open_dataset(f"{config.obs_data_folder}/siconca_EASE.nc") as ds:
            da = next(iter(ds.data_vars.values()))

    valid_dates = [pd.Timestamp(date) for date in da.time.values]

    input_dates = [forecast_month - pd.DateOffset(years=1+lag) for lag in range(n_linear_years)]
    input_dates

    # Do not use missing months in the linear trend projection
    input_dates = [date for date in input_dates if date not in config.missing_dates]

    # Chop off input date from before data start
    input_dates = [date for date in input_dates if date in valid_dates]

    input_dates = sorted(input_dates)

    # The actual number of past years used
    actual_n_linear_years = len(input_dates)

    da = da.sel(time=input_dates)

    input_maps = np.array(da.data)

    x = np.arange(actual_n_linear_years)
    y = input_maps.reshape(actual_n_linear_years, -1)

    # Fit the least squares linear coefficients
    r = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0]

    # y = mx + c
    output_map = np.matmul(np.array([actual_n_linear_years, 1]), r).reshape(432, 432)

    land_mask_path = os.path.join(config.mask_data_folder, config.land_mask_filename)
    land_mask = np.load(land_mask_path)
    output_map[land_mask] = 0.

    output_map[output_map < 0] = 0.
    output_map[output_map > 1] = 1.

    sie = np.sum(output_map > 0.15) * 25**2

    return output_map, sie


def train_icenet(args):
    """
    Train IceNet using the arguments specified in the `args` namespace.
    :param args: Namespace of configuration parameters
    """
    # init
    pl.seed_everything(args.seed)
    
    # configure datasets and dataloaders
    dataloader_config = os.path.join("/data/hpcdata/users/anddon76/icenet/icenet-experimental", args.dataloader_config)
    train_dataset = IceNetDataSetPyTorch(dataloader_config, mode="train", batch_size=args.batch_size, shuffling=False)
    val_dataset = IceNetDataSetPyTorch(dataloader_config, mode="val", batch_size=args.batch_size, shuffling=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  persistent_workers=True if args.num_workers > 0 else False, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                persistent_workers=True if args.num_workers > 0 else False, shuffle=False)

    # configure PyTorch Lightning module
    lit_module = LitDiffusion(learning_rate=args.learning_rate)
            
    # set up wandb logging
    wandb.init(project="icenet-diffusion")
    if args.name != "default":
        wandb.run.name = args.name
    wandb_logger = pl.loggers.WandbLogger(project="icenet-diffusion")
    wandb_logger.experiment.config.update(args)

    # comment/uncomment the following line to track gradients
    # note that wandb cannot parallelise across multiple gpus when tracking gradients
    # wandb_logger.watch(model, log="all", log_freq=10)

    # set up trainer configuration
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run
    )
    trainer.logger = wandb_logger

    # train model
    print(f"Training {len(train_dataset)} examples / {len(train_dataloader)} batches (batch size {args.batch_size}).")
    print(f"Validating {len(val_dataset)} examples / {len(val_dataloader)} batches (batch size {args.batch_size}).")
    print(f"All arguments: {args}")
    trainer.fit(lit_module, train_dataloader, val_dataloader)


if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser(description="Train IceNet")
    parser.add_argument("--name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--model", default="gan", type=str, choices=["unet", "gan"],
                        help="Choice of model architecture", required=False)
    
    # model configurations applicable to both UNet and GAN
    parser.add_argument("--dataloader_config", default="dataset_config.single_year.json", type=str,
                        help="Filename of dataloader_config.json file")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    # hardware configurations applicable to both UNet and GAN
    parser.add_argument("--accelerator", default="auto", type=str, help="PytorchLightning training accelerator")
    parser.add_argument("--devices", default=1, type=int, help="PytorchLightning number of devices to run on")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of workers in dataloader")
    parser.add_argument("--precision", default=32, type=int, choices=[32, 16], help="Precision for training")
    
    # logging configurations applicable to both UNet and GAN
    parser.add_argument("--log_every_n_steps", default=10, type=int, help="How often to log during training")
    parser.add_argument("--max_epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--num_sanity_val_steps", default=1, type=int, 
                        help="Number of batches to sanity check before training")
    parser.add_argument("--limit_train_batches", default=1.0, type=float, help="Proportion of training dataset to use")
    parser.add_argument("--limit_val_batches", default=1.0, type=float, help="Proportion of validation dataset to use")
    parser.add_argument("--n_to_visualise", default=1, type=int, help="How many forecasts to visualise")
    parser.add_argument("--fast_dev_run", default=False, type=eval, help="Whether to conduct a fast one-batch dev run")

    # let's go
    args = parser.parse_args()
    train_icenet(args)

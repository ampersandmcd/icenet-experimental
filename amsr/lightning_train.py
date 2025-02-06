import os
import sys
import pandas as pd
import lightning.pytorch as pl
import wandb
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.ops.focal_loss import sigmoid_focal_loss
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, HeunDiscreteScheduler
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from utils import BinaryAccuracy, SIEError, SIAError, RMSError, MAError, compute_spectrum2d
from utils import IceNetDataSetPyTorch, IterableIceNetDataSetPyTorch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchvision.transforms import Normalize

# trade off speed and performance depending on gpu
torch.set_float32_matmul_precision("medium")
# torch.set_float32_matmul_precision("high")

class DatasetAMSR(Dataset):
    def __init__(self,
                 path: str,
                 start: str,
                 end: str,
                 normalise: bool = False
                ):
        files = sorted(os.listdir(path))
        start_idx = files.index(f"{start}.npy")
        end_idx = files.index(f"{end}.npy")
        self._path = path
        self._files = files[start_idx:end_idx]
        self._start = start
        self._end = end
        self._stats = pd.read_csv("/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera_stats.csv")
        self._xmean = self._stats["Mean"][1:].values
        self._xstd = self._stats["Std"][1:].values
        # self._xnorm = Normalize(self._xmean, self._xstd)
        self._xnorm = Normalize([0.5], [0.5])
        self._ymean = self._stats["Mean"][[0]].values
        self._ystd = self._stats["Mean"][[0]].values
        # self._ynorm = Normalize(self._ymean, self._ystd)
        self._ynorm = Normalize([0.5], [0.5])
        self._normalise = normalise
    
    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, idx):
        file = self._files[idx]
        sample = np.load(os.path.join(self._path, file))
        # note from dataset creation sample = np.array([ice_t, ice_t_1, ice_t_2, clim, land, cos, sin])
        x = sample[1:]  # hence x is [ice_t_1, ice_t_2, clim, land, cos, sin]
        y = sample[[0]]  # hence y is [ice_t] (keep first dim for downstream convenience)
        mask = sample[[4]]  # hence mask is [land] (keep first dim for downstream convenience)
        date = file[:-4]
        x, y, mask = torch.from_numpy(x).to(torch.float), torch.from_numpy(y).to(torch.float), torch.from_numpy(mask).to(torch.float)
        x, y, mask = torch.nan_to_num(x), torch.nan_to_num(y), torch.nan_to_num(mask)  # set nan to 0, reconsider later
        if self._normalise:
            x = self._xnorm(x)
            y = self._ynorm(y)
        return x, y, mask, date


class OverfitDatasetAMSR(DatasetAMSR):
    """A class for overfitting on a single AMSR sample"""
    def __init__(self,
                 path: str,
                 start: str,
                 end: str,
                 replicates: int  # how many times to replicate the subset in [start, end] to be overfit upon
                ):
        super().__init__(path, start, end)
        self._files = self._files * replicates


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
                 resolution=128,
                 in_channels=6,
                 out_channels=1,
                 block_out_channels=(32, 64, 128, 256),
                 north=False,
                 cos_path="/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/ice_only_lag_two/meta/south/cos/cos.nc",
                 sin_path="/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/ice_only_lag_two/meta/south/sin/sin.nc",
                 linear_trend_path="/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/ice_only_lag_two/osisaf/south/siconca/siconca_linear_trend.nc",
                 ground_truth_path="/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/ice_only_lag_two/osisaf/south/siconca/siconca_abs.nc",
                 forecast_residuals=False,
                 num_train_timesteps=10,
                 num_inference_timesteps=10,
                 num_samples=3,
                 noise_scheduler=DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2"),
                 prediction_type="v_prediction",
                 min_sic_threshold=0,
                 num_rollout_steps=3,
                 learning_rate=1e-3):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.north = north
        self.cos_path = cos_path
        self.sin_path = sin_path
        self.linear_trend_path = linear_trend_path
        self.ground_truth_path = ground_truth_path
        self.forecast_residuals = forecast_residuals
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.num_samples = num_samples
        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type
        self.min_sic_threshold = min_sic_threshold
        self.num_rollout_steps = num_rollout_steps
        self.learning_rate = learning_rate
        self.model = UNet2DModel(
            sample_size=resolution,  # the target image resolution
            in_channels=in_channels + out_channels,  # input channels + noise (target) channels
            out_channels=out_channels,  # target channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=block_out_channels,
            down_block_types=(
                ["DownBlock2D"]*4
            ),
            up_block_types=(
                ["UpBlock2D"]*4
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
        val_rollout_metrics = {
            "val_rollout_rmse": RMSError(leadtimes_to_evaluate=list(range(num_rollout_steps))),
            "val_rollout_mae": MAError(leadtimes_to_evaluate=list(range(num_rollout_steps))),
            "val_rollout_accuracy": BinaryAccuracy(leadtimes_to_evaluate=list(range(num_rollout_steps))),
            "val_rollout_sieerror": SIEError(leadtimes_to_evaluate=list(range(num_rollout_steps))),
            "val_rollout_siaerror": SIAError(leadtimes_to_evaluate=list(range(num_rollout_steps)))
        }
        self.val_rollout_metrics = MetricCollection(val_rollout_metrics)
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
        # net_input = self.noise_scheduler.scale_model_input(net_input, t)
        return self.model(net_input, t).sample  # (bs, target_c, resolution, resolution)
    
    def rollout(self, x, sample_weight, start_date, end_date):
        """
        Perform an autoregressive rollout forecast.
        """
        x = x[[0]].repeat(self.num_samples, 1, 1, 1).to(self.device)  # copy same x conditions to num_samples size batch
        mask = torch.where(sample_weight[0] > 0, 1, 0).to(self.device)
        date_range = pd.date_range(start_date, end_date)
        forecast = []

        # iterate over days
        for date in tqdm(date_range):

            # sample rollout
            noisy_y = torch.randn(self.num_samples, self.out_channels,
                                  self.resolution, self.resolution).to(self.device)
            for t in self.noise_scheduler.timesteps:
                with torch.no_grad():
                    pred = self.forward(noisy_y, t, x)
                noisy_y = self.noise_scheduler.step(pred, t, noisy_y).prev_sample
            
            if self.forecast_residuals:
                noisy_y = noisy_y + x[:, [0]] # add on most recent ice condition (b, t, h, w) to THIS PREDICTION

            # post process sample
            noisy_y = noisy_y.clip(0, 1)  # TODO reconsider clipping SIC is bounded between 0 and 1
            noisy_y = noisy_y * mask  # mask out land and midlatitude ocean
            # noisy_y = torch.where(noisy_y < self.min_sic_threshold, 0, noisy_y)  # clip residual noise DONT DO THIS FOR range -1, +1
            forecast.append(noisy_y)

            # get next step's contextual predictors (without "cheating" i.e. we discard ice info)
            next_date = date + pd.Timedelta(1, "day")
            next_f = f"/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera/{next_date.strftime('%Y%m%d')}.npy"
            next_sample = np.load(next_f)
            next_context = torch.from_numpy(next_sample[3:]).nan_to_num().to(torch.float).to(self.device)
            next_context = next_context.repeat(self.num_samples, 1, 1, 1)  # broadcast copies for ensemble

            # note sample = [ice_t, ice_t_1, ice_t_2, clim, land, cos, sin]            
            x = torch.cat((
                noisy_y,   # set ice_t -> ice_t_1 from prediction
                x[:, [0]],  # set ice_t_1 -> ice_t_2 from last predictor
                next_context, # climatology, land, and cos/sin embeddings for next day without
            ), dim=1)  # update to next timestep, t is first, t-1 is second
            x = x.to(dtype=torch.float32)
            
        # concatenate into forecast
        forecast = torch.cat(forecast, dim=1).to(self.device)

        # compile ground truth and climatology
        # note from dataset creation sample = np.array([ice_t, ice_t_1, ice_t_2, clim, land, cos, sin])
        truth, clim = [], []
        date = start_date
        while date <= end_date:
            f = f"/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera/{date.strftime('%Y%m%d')}.npy"
            sample = np.load(f)
            truth.append(sample[0])  # ice_t is first element of sample
            clim.append(sample[3])  # ice_t is first element of sample
            date = date + pd.Timedelta(1, "day")
        truth = torch.from_numpy(np.array(truth)).nan_to_num().to(torch.float).to(self.device)
        clim = torch.from_numpy(np.array(clim)).nan_to_num().to(torch.float).to(self.device)

        return forecast, clim, truth

    def training_step(self, batch, batch_idx: int):
        """
        Perform a pass through a batch of training data.
        :param batch: Batch of input, output, weight triplets.
        :param batch_idx: Index of batch.
        :return: Loss from this batch of data for use in backprop.
        """
        x, y, sample_weight, date = batch
        if self.forecast_residuals:
            y = y - x[:, [0], :, :]  # subtract away most recent ice condition (b, t, h, w)
        noise = torch.randn_like(y)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x.shape[0],)).long().to(self.device)
        noisy_y = self.noise_scheduler.add_noise(y, noise, timesteps)
        pred = self.forward(noisy_y, timesteps, x)

        # TODO: mask this loss
        # TODO: look at sampling code wrt epsilon and see why not working, loss is converging but preds are bad

        if self.prediction_type == "epsilon":
            loss = F.mse_loss(pred, noise)
        elif self.prediction_type == "sample":
            loss = F.mse_loss(pred, y)
        elif self.prediction_type == "v_prediction":
            # copied from DDPMScheduler.get_velocity
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=y.device, dtype=y.dtype)
            sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(y.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(y.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * y
            # end copy
            loss = F.mse_loss(pred, velocity)

        else:
            raise NotImplementedError(f"Not a valid prediction type: {self.prediction_type}")

        self.log("train_loss", loss, prog_bar=True)
        if torch.isnan(loss):
            # catch nan loss
            return None
        return loss

    def validation_step(self, batch, batch_idx):

        # limit validation to one batch for now
        if batch_idx != 0:
            return

        x, y, sample_weight, dates = batch
        if self.forecast_residuals:
            y = y - x[:, [0], :, :]  # subtract away most recent ice condition (b, t, h, w)
        
        # do batches one at a time, otherwise likely won't fit on GPU
        bs = x.shape[0]
        mask = torch.where(sample_weight[0] > 0, 1, 0).to(self.device)
        samples = []
        for i in range(self.num_samples):  # for each condition in batch, generate multiple samples

            noisy_y = torch.randn(bs, self.out_channels,
                                  self.resolution, self.resolution).to(self.device)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for t in tqdm(self.noise_scheduler.timesteps):
                with torch.no_grad():
                    pred = self.forward(noisy_y, t, x)
                noisy_y = self.noise_scheduler.step(pred, t, noisy_y).prev_sample

            if self.forecast_residuals:
                noisy_y = noisy_y + x[:, [0], :, :]  # add on most recent ice condition (b, t, h, w) to THIS PREDICTION

            samples.append(noisy_y)

        if self.forecast_residuals:
            y = y + x[:, [0], :, :]  # add on most recent ice condition (b, t, h, w) to GROUND TRUTH
        samples = torch.stack(samples)  # shape (num_samples, batch_size, timesteps, y, x)
        # samples = samples.clip(0, 1)  # TODO re-evaluate clipping
        samples = samples * mask
        means = samples.mean(dim=0)  # take ensemble mean across num_samples dimension
        
        # compute metrics 
        val_mse = F.mse_loss(means, y)
        self.log("val_mse", val_mse, prog_bar=True)
        self.val_metrics.update(means, y, sample_weight)

        # dig deeper for the first init date
        if batch_idx == 0:
            
            # visualise sample
            stds = samples.std(dim=0)  # take ensemble std across num_samples dimension
            diffs = y - means
            nrows = min(bs, 4)
            ncols = self.num_samples + 6
            fig, ax = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
            for i in range(nrows):
                for j in range(self.num_samples):
                    ax[i][j].imshow(samples[j][i].squeeze().cpu(), vmin=0, vmax=1)
                    ax[i][j].set_title(f"Init {dates[i]} Sample {j+1}")
                    ax[i][self.num_samples + 4].hist(samples[j][i].cpu().ravel(), bins=100,
                                                     histtype=u'step', log=True, label=f"Sample {j+1}")
                    kvals, Abins = compute_spectrum2d(samples[j][i].cpu().squeeze())
                    ax[i][self.num_samples + 5].loglog(kvals, Abins, label=f"Sample {j+1}")
                ax[i][self.num_samples].imshow(means[i].squeeze().cpu(), vmin=0, vmax=1)
                ax[i][self.num_samples].set_title(f"Mean for Init {dates[i]}")
                ax[i][self.num_samples + 1].imshow(stds[i].squeeze().cpu(), cmap="cividis")
                ax[i][self.num_samples + 1].set_title(f"Std for Init {dates[i]}")
                ax[i][self.num_samples + 2].imshow(y[i].squeeze().cpu(), vmin=0, vmax=1)
                ax[i][self.num_samples + 2].set_title(f"Truth for Init {dates[i]}")
                ax[i][self.num_samples + 3].imshow(diffs[i].squeeze().cpu(), cmap="bwr", vmin=-1, vmax=1)
                ax[i][self.num_samples + 3].set_title(f"(Truth - Mean) for Init {dates[i]}")

                ax[i][self.num_samples + 4].hist(means[i].cpu().ravel(), bins=100,
                                                 histtype=u'step', log=True, label=f"Ensemble Mean")
                ax[i][self.num_samples + 4].hist(y[i].cpu().ravel(), bins=100,
                                                 histtype=u'step', log=True, label=f"Truth")
                ax[i][self.num_samples + 4].set_title(f"Pixel Historgram for Init {dates[i]}")

                kvals, Abins = compute_spectrum2d(means[i].cpu().squeeze())
                ax[i][self.num_samples + 5].loglog(kvals, Abins, label=f"Ensemble Mean")  
                kvals, Abins = compute_spectrum2d(y[i].cpu().squeeze())
                ax[i][self.num_samples + 5].loglog(kvals, Abins, label=f"Truth")            
                ax[i][self.num_samples + 5].set_title(f"Spectrum for Init {dates[i]}")

            plt.legend()
            plt.tight_layout()
            self.logger.experiment.log({"Sample": plt})
            plt.close()

            # rollout
            start_date = pd.to_datetime(dates[0])
            end_date = start_date + pd.Timedelta(self.num_rollout_steps - 1, "D")
            try:
                forecast, trend, truth = self.rollout(x[[0]], sample_weight, start_date, end_date)

                # create imshow video
                fps = 3
                fig, ax = plt.subplots(2, 4, figsize=(16, 8))
                ax = ax.ravel()
                def img_update(step):
                    fc_1.set_data(forecast[0].squeeze()[step].cpu().numpy())
                    fc_2.set_data(forecast[1].squeeze()[step].cpu().numpy())
                    fc_3.set_data(forecast[2].squeeze()[step].cpu().numpy())
                    fc_std.set_data(forecast.std(dim=0).squeeze()[step].cpu().numpy())
                    fc_mean.set_data(forecast.mean(dim=0).squeeze()[step].cpu().numpy())
                    fc_trend.set_data(trend[step].squeeze().cpu().numpy())
                    fc_true.set_data(truth[step].squeeze().cpu().numpy())
                    fc_diff.set_data((truth[step].squeeze() - forecast.mean(dim=0)[step]).cpu().numpy())
                    title.set_text(f"Rollout Step {step+1} | Init {start_date} | Valid {start_date + pd.Timedelta(step+1, 'D')}")

                fc_1 = ax[0].imshow(forecast[0].squeeze()[0].cpu().numpy(), vmin=0, vmax=1, animated=True)
                ax[0].set_title("Ensemble Member 1")
                fc_2 = ax[1].imshow(forecast[1].squeeze()[0].cpu().numpy(), vmin=0, vmax=1, animated=True)
                ax[1].set_title("Ensemble Member 2")
                fc_3 = ax[2].imshow(forecast[2].squeeze()[0].cpu().numpy(), vmin=0, vmax=1, animated=True)
                ax[2].set_title("Ensemble Member 3")
                fc_std = ax[3].imshow(forecast.std(dim=0).squeeze()[0].cpu().numpy(), cmap="cividis", animated=True)
                ax[3].set_title("Ensemble Std")
                fc_mean = ax[4].imshow(forecast.mean(dim=0).squeeze()[0].cpu().numpy(), vmin=0, vmax=1, animated=True)
                ax[4].set_title("Ensemble Mean")
                fc_trend = ax[5].imshow(trend[0].squeeze().cpu().numpy(), vmin=0, vmax=1, animated=True)
                ax[5].set_title("Linear Trend Forecast")
                fc_true = ax[6].imshow(truth[0].squeeze().cpu().numpy(), vmin=0, vmax=1, animated=True)
                ax[6].set_title("Truth")
                fc_diff = ax[7].imshow((truth[0].squeeze() - forecast.mean(dim=0)[0]).cpu().numpy(), vmin=-1, vmax=1, cmap="bwr", animated=True)
                ax[7].set_title("(Truth - Ensemble Mean)")
                title = plt.suptitle(f"Rollout Step 1 | Init {start_date} | Valid {start_date + pd.Timedelta(1, 'D')}")
                plt.tight_layout()
                animation = FuncAnimation(fig, img_update, range(self.num_rollout_steps), interval=1000 / fps)
                plt.close()
                animation.save("temp.mp4", fps=fps)
                wandb.log({"rollout": wandb.Video("temp.mp4")})

                # create histogram video
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    def update_hist(step):
                        # a bit hacky since we are using histtype="step"
                        for d, hist in zip(data, hists):
                            n, _ = np.histogram(d.squeeze()[step].cpu().numpy().ravel(), 100)
                            heights = np.repeat(n, 2)
                            hist[0].xy[1:-1, 1] = heights  # update heights of all but first and last entry
                        title.set_text(f"Rollout Step {step+1} | Init {start_date} | Valid {start_date + pd.Timedelta(step+1, 'D')}")

                    _, _, fc_hist_1 = ax.hist(forecast[0].squeeze()[0].cpu().numpy().ravel(), bins=100,
                                            histtype=u'step', log=True, label=f"Ensemble Member 1")
                    _, _, fc_hist_2 = ax.hist(forecast[1].squeeze()[0].cpu().numpy().ravel(), bins=100,
                                            histtype=u'step', log=True, label=f"Ensemble Member 2")
                    _, _, fc_hist_3 = ax.hist(forecast[2].squeeze()[0].cpu().numpy().ravel(), bins=100,
                                            histtype=u'step', log=True, label=f"Ensemble Member 3")
                    _, _, fc_hist_mean = ax.hist(forecast.mean(dim=0).squeeze()[0].cpu().numpy().ravel(), bins=100,
                                            histtype=u'step', log=True, label=f"Ensemble Mean")
                    _, _, fc_hist_trend = ax.hist(trend.squeeze()[0].cpu().numpy().ravel(), bins=100,
                                            histtype=u'step', log=True, label=f"Linear Trend Forecast")
                    _, _, fc_hist_truth = ax.hist(truth.squeeze()[0].cpu().numpy().ravel(), bins=100,
                                                histtype=u'step', log=True, label=f"Truth")
                    data = [forecast[0], forecast[1], forecast[2], forecast.mean(dim=0), trend, truth]
                    hists = [fc_hist_1, fc_hist_2, fc_hist_3, fc_hist_mean, fc_hist_trend, fc_hist_truth]
                    title = plt.suptitle(f"Rollout Step 1 | Init {start_date} | Valid {start_date + pd.Timedelta(1, 'D')}")
                    plt.legend()
                    plt.tight_layout()
                    animation = FuncAnimation(fig, update_hist, range(self.num_rollout_steps), interval=1000 / fps)
                    plt.close()
                    animation.save("temp.mp4", fps=fps)
                    wandb.log({"rollout_hist": wandb.Video("temp.mp4")})
                except:
                    print(f"Rollout histogram failed on epoch {self.current_epoch}")

                # create spectrum video
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    def update_spectrum(step):
                        for d, line in zip(data, lines):
                            k, A = compute_spectrum2d(d.squeeze()[step].cpu().numpy())
                            line.set_ydata(A)
                        title.set_text(f"Rollout Step {step+1} | Init {start_date} | Valid {start_date + pd.Timedelta(step+1, 'D')}")

                    k, A = compute_spectrum2d(forecast[0].squeeze()[0].cpu().numpy())
                    fc_line_1, = ax.loglog(k, A, label=f"Ensemble Member 1", animated=True)
                    k, A = compute_spectrum2d(forecast[1].squeeze()[0].cpu().numpy())
                    fc_line_2, = ax.loglog(k, A, label=f"Ensemble Member 2", animated=True)
                    k, A = compute_spectrum2d(forecast[2].squeeze()[0].cpu().numpy())
                    fc_line_3, = ax.loglog(k, A, label=f"Ensemble Member 3", animated=True)
                    k, A = compute_spectrum2d(forecast.mean(dim=0).squeeze()[0].cpu().numpy())
                    fc_line_mean, = ax.loglog(k, A, label=f"Ensemble Mean", animated=True)
                    k, A = compute_spectrum2d(trend.squeeze()[0].cpu().numpy())
                    fc_line_trend, = ax.loglog(k, A, label=f"Linear Trend Forecast", animated=True)
                    k, A = compute_spectrum2d(truth.squeeze()[0].cpu().numpy())
                    fc_line_truth, = ax.loglog(k, A, label=f"Truth", animated=True)
                    data = [forecast[0], forecast[1], forecast[2], forecast.mean(dim=0), trend, truth]
                    lines = [fc_line_1, fc_line_2, fc_line_3, fc_line_mean, fc_line_trend, fc_line_truth]
                    title = plt.suptitle(f"Rollout Step 1 | Init {start_date} | Valid {start_date + pd.Timedelta(1, 'D')}")
                    plt.xlabel("Wavenumber (Spatial Frequency)")
                    plt.ylabel("Signal Power")
                    plt.legend()
                    plt.tight_layout()
                    animation = FuncAnimation(fig, update_spectrum, range(self.num_rollout_steps), interval=1000 / fps)
                    plt.close()
                    animation.save("temp.mp4", fps=fps)
                    wandb.log({"rollout_spectrum": wandb.Video("temp.mp4")})
                except:
                    print(f"Rollout histogram failed on epoch {self.current_epoch}")

                # log rollout metrics using ensemble mean
                rollout_preds = forecast.mean(dim=0).unsqueeze(0)
                rollout_target = truth.unsqueeze(0)
                rollout_sw = sample_weight[[0]].repeat(1, self.num_rollout_steps, 1, 1)
                self.val_rollout_metrics.update(rollout_preds, rollout_target, rollout_sw)
                self.log_dict(self.val_rollout_metrics.compute())
                self.val_rollout_metrics.reset()
            except:
                print(f"Rollout failed on epoch {self.current_epoch} with init date {start_date}")
    
    def on_validation_epoch_start(self):
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.val_metrics.reset()
        self.noise_scheduler.set_timesteps(self.num_train_timesteps)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.test_metrics.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.CosineAnnealingLR(opt, 100, T_mult=1, eta_min=1e-5)
        return [opt], [] # [scheduler]  # add schedulers to second list if desired
    

class LitUNet(pl.LightningModule):
    """
    A LightningModule wrapping the UNet implementation of IceNet.
    :param resolution: size of the forecast field
    :param in_channels: number of channels coming into the network = predictors + noise
    :param out_channels: number of channels coming out of the network = forecast steps at once
    :param block_out_channels: channels per UNet block, rough control over network size
    :param forecast_residuals: if True, forecasts ice directly, if False, forecasts residuals between steps
    :param noise_scheduler: HF diffusers noise scheduler object
    """
    def __init__(self,
                 resolution=128,
                 in_channels=6,
                 out_channels=1,
                 block_out_channels=(32, 64, 128, 256),
                 north=False,
                 cos_path="/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/ice_only_lag_two/meta/south/cos/cos.nc",
                 sin_path="/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/ice_only_lag_two/meta/south/sin/sin.nc",
                 linear_trend_path="/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/ice_only_lag_two/osisaf/south/siconca/siconca_linear_trend.nc",
                 ground_truth_path="/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/ice_only_lag_two/osisaf/south/siconca/siconca_abs.nc",
                 forecast_residuals=False,
                 num_samples=1,
                 min_sic_threshold=0,
                 num_rollout_steps=10,
                 learning_rate=1e-3):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.north = north
        self.cos_path = cos_path
        self.sin_path = sin_path
        self.linear_trend_path = linear_trend_path
        self.ground_truth_path = ground_truth_path
        self.forecast_residuals = forecast_residuals
        self.num_samples = num_samples
        self.min_sic_threshold = min_sic_threshold
        self.num_rollout_steps = num_rollout_steps
        self.learning_rate = learning_rate
        self.model = UNet2DModel(
            sample_size=resolution,  # the target image resolution
            in_channels=in_channels,  # input channels only
            out_channels=out_channels,  # target channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=block_out_channels,
            down_block_types=(
                ["DownBlock2D"]*4
            ),
            up_block_types=(
                ["UpBlock2D"]*4
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
        val_rollout_metrics = {
            "val_rollout_rmse": RMSError(leadtimes_to_evaluate=list(range(num_rollout_steps))),
            "val_rollout_mae": MAError(leadtimes_to_evaluate=list(range(num_rollout_steps))),
            "val_rollout_accuracy": BinaryAccuracy(leadtimes_to_evaluate=list(range(num_rollout_steps))),
            "val_rollout_sieerror": SIEError(leadtimes_to_evaluate=list(range(num_rollout_steps))),
            "val_rollout_siaerror": SIAError(leadtimes_to_evaluate=list(range(num_rollout_steps)))
        }
        self.val_rollout_metrics = MetricCollection(val_rollout_metrics)
        test_metrics = {
            "val_rmse": RMSError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_mae": MAError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_accuracy": BinaryAccuracy(leadtimes_to_evaluate=list(range(out_channels))),
            "val_sieerror": SIEError(leadtimes_to_evaluate=list(range(out_channels))),
            "val_siaerror": SIAError(leadtimes_to_evaluate=list(range(out_channels)))
        }
        self.test_metrics = MetricCollection(test_metrics)

        self.save_hyperparameters()

    def forward(self, condition_x):
        """
        Implement forward function.
        :param condition_x: Conditioning information.
        :return: Outputs of model.
        """
        return self.model(condition_x, 0).sample  # (bs, target_c, resolution, resolution) (ignore timestep)
    
    def rollout(self, x, sample_weight, start_date, end_date):
        """
        Perform an autoregressive rollout forecast.
        """
        x = x[[0]].repeat(self.num_samples, 1, 1, 1).to(self.device)  # copy same x conditions to num_samples size batch
        mask = torch.where(sample_weight[0] > 0, 1, 0).to(self.device)
        date_range = pd.date_range(start_date, end_date)
        forecast = []

        # iterate over days
        for date in tqdm(date_range):

            # sample rollout
            with torch.no_grad():
                pred = self.forward(x)
            
            if self.forecast_residuals:
                pred = pred + x[:, [0]] # add on most recent ice condition (b, t, h, w) to THIS PREDICTION

            # post process sample
            pred = pred.clip(0, 1)  # SIC is bounded between 0 and 1
            pred = pred * mask  # mask out land and midlatitude ocean
            pred = torch.where(pred < self.min_sic_threshold, 0, pred)  # clip residual noise
            forecast.append(pred)

            # update date embeddings
            next_date = date + pd.Timedelta(1, "day")
            linear_trend = xr.open_dataarray(self.linear_trend_path).sel(time=next_date).to_numpy()
            linear_trend = torch.from_numpy(linear_trend).to(self.device) * torch.ones_like(x[:, [0]])
            trig_date = pd.to_datetime(f"2012-{next_date.month}-{next_date.day}")
            cos = xr.open_dataarray(self.cos_path).sel(time=trig_date).to_numpy()
            cos = torch.from_numpy(cos).to(self.device) * torch.ones_like(x[:, [0]])
            sin = xr.open_dataarray(self.sin_path).sel(time=trig_date).to_numpy()
            sin = torch.from_numpy(sin).to(self.device) * torch.ones_like(x[:, [0]])
            
            # create next step's predictor                
            x = torch.cat((
                pred,   # set ice at t+1 to ice at t
                x[:, [0]],  # set ice at t to ice at t-1
                linear_trend, # linear climatology forecast
                cos,  # new cos embedding
                x[:, [4]],  # maintain land mask
                sin,  # new sin embedding
            ), dim=1)  # update to next timestep, t is first, t-1 is second
            x = x.to(dtype=torch.float32)
            
        # concatenate into forecast
        forecast = torch.cat(forecast, dim=1).to(self.device)

        # compile ground truth
        truth = xr.open_dataarray(self.ground_truth_path).sel(time=date_range).to_numpy()
        truth = torch.from_numpy(truth).to(self.device)

        # compile linear trend
        trend = xr.open_dataarray(self.linear_trend_path).sel(time=date_range).to_numpy()
        trend = torch.from_numpy(trend).to(self.device)

        return forecast, trend, truth

    def training_step(self, batch, batch_idx: int):
        """
        Perform a pass through a batch of training data.
        :param batch: Batch of input, output, weight triplets.
        :param batch_idx: Index of batch.
        :return: Loss from this batch of data for use in backprop.
        """
        x, y, sample_weight, date = batch
        if self.forecast_residuals:
            y = y - x[:, [0], :, :]  # subtract away most recent ice condition (b, t, h, w)
        pred = self.forward(x)
        loss = F.mse_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        if torch.isnan(loss):
            # catch nan loss
            return None
        return loss

    def validation_step(self, batch, batch_idx):

        x, y, sample_weight, dates = batch
        if self.forecast_residuals:
            y = y - x[:, [0], :, :]  # subtract away most recent ice condition (b, t, h, w)
        
        # do batches one at a time, otherwise likely won't fit on GPU
        bs = x.shape[0]
        mask = torch.where(sample_weight[0] > 0, 1, 0).to(self.device)
        samples = []
        for i in range(self.num_samples):  # for each condition in batch, generate multiple samples

            with torch.no_grad():
                pred = self.forward(x)
            
            if self.forecast_residuals:
                pred = pred + x[:, [0]] # add on most recent ice condition (b, t, h, w) to THIS PREDICTION

            samples.append(pred)

        if self.forecast_residuals:
            y = y + x[:, [0], :, :]  # add on most recent ice condition (b, t, h, w) to GROUND TRUTH

        samples = torch.stack(samples)  # shape (num_samples, batch_size, timesteps, y, x)
        samples = samples.clip(0, 1)
        samples = samples * mask
        means = samples.mean(dim=0)  # take ensemble mean across num_samples dimension
        
        # compute metrics 
        val_mse = F.mse_loss(means, y)
        self.log("val_mse", val_mse, prog_bar=True)
        self.val_metrics.update(means, y, sample_weight)

        # dig deeper for the first init date
        if batch_idx == 0:
            
            # visualise sample
            stds = samples.std(dim=0)  # take ensemble std across num_samples dimension
            diffs = y - means
            nrows, ncols = bs, self.num_samples + 6
            fig, ax = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
            for i in range(nrows):
                for j in range(self.num_samples):
                    ax[i][j].imshow(samples[j][i].squeeze().cpu())
                    ax[i][j].set_title(f"Init {dates[i]} Sample {j+1}")
                    ax[i][self.num_samples + 4].hist(samples[j][i].cpu().ravel(), bins=100,
                                                     histtype=u'step', log=True, label=f"Sample {j+1}")
                    kvals, Abins = compute_spectrum2d(samples[j][i].cpu().squeeze())
                    ax[i][self.num_samples + 5].loglog(kvals, Abins, label=f"Sample {j+1}")
                ax[i][self.num_samples].imshow(means[i].squeeze().cpu())
                ax[i][self.num_samples].set_title(f"UNet for Init {dates[i]}")
                ax[i][self.num_samples + 1].imshow(y[i].squeeze().cpu())
                ax[i][self.num_samples + 1].set_title(f"Truth for Init {dates[i]}")
                ax[i][self.num_samples + 2].imshow(diffs[i].squeeze().cpu(), cmap="bwr", vmin=-1, vmax=1)
                ax[i][self.num_samples + 2].set_title(f"(Truth - UNet) for Init {dates[i]}")

                ax[i][self.num_samples + 3].hist(means[i].cpu().ravel(), bins=100,
                                                 histtype=u'step', log=True, label=f"Ensemble Mean")
                ax[i][self.num_samples + 3].hist(y[i].cpu().ravel(), bins=100,
                                                 histtype=u'step', log=True, label=f"Truth")
                ax[i][self.num_samples + 3].set_title(f"Pixel Historgram for Init {dates[i]}")

                kvals, Abins = compute_spectrum2d(means[i].cpu().squeeze())
                ax[i][self.num_samples + 4].loglog(kvals, Abins, label=f"UNet")  
                kvals, Abins = compute_spectrum2d(y[i].cpu().squeeze())
                ax[i][self.num_samples + 4].loglog(kvals, Abins, label=f"Truth")            
                ax[i][self.num_samples + 4].set_title(f"Spectrum for Init {dates[i]}")

            plt.legend()
            plt.tight_layout()
            self.logger.experiment.log({"Sample": plt})
            plt.close()

            # rollout
            start_date = pd.to_datetime(dates[0])
            end_date = start_date + pd.Timedelta(self.num_rollout_steps - 1, "D")
            forecast, trend, truth = self.rollout(x[[0]], sample_weight, start_date, end_date)

            # create imshow video
            fps = 5
            fig, ax = plt.subplots(2, 4, figsize=(16, 8))
            ax = ax.ravel()
            def img_update(step):
                fc_mean.set_data(forecast.mean(dim=0).squeeze()[step].cpu().numpy())
                fc_trend.set_data(trend[step].squeeze().cpu().numpy())
                fc_true.set_data(truth[step].squeeze().cpu().numpy())
                fc_diff.set_data((truth[step].squeeze() - forecast.mean(dim=0)[step]).cpu().numpy())
                title.set_text(f"Rollout Step {step+1} | Init {start_date} | Valid {start_date + pd.Timedelta(step+1, 'D')}")

            fc_mean = ax[0].imshow(forecast.mean(dim=0).squeeze()[0].cpu().numpy(), vmin=0, vmax=1, animated=True)
            ax[0].set_title("UNet")
            fc_trend = ax[1].imshow(trend[0].squeeze().cpu().numpy(), vmin=0, vmax=1, animated=True)
            ax[1].set_title("Linear Trend Forecast")
            fc_true = ax[2].imshow(truth[0].squeeze().cpu().numpy(), vmin=0, vmax=1, animated=True)
            ax[2].set_title("Truth")
            fc_diff = ax[3].imshow((truth[0].squeeze() - forecast.mean(dim=0)[0]).cpu().numpy(), vmin=-1, vmax=1, cmap="bwr", animated=True)
            ax[3].set_title("(Truth - UNet)")
            title = plt.suptitle(f"Rollout Step 1 | Init {start_date} | Valid {start_date + pd.Timedelta(1, 'D')}")
            plt.tight_layout()
            animation = FuncAnimation(fig, img_update, range(self.num_rollout_steps), interval=1000 / fps)
            plt.close()
            animation.save("temp.mp4", fps=fps)
            wandb.log({"rollout": wandb.Video("temp.mp4")})

            # create histogram video
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            def update_hist(step):
                # a bit hacky since we are using histtype="step"
                for d, hist in zip(data, hists):
                    n, _ = np.histogram(d.squeeze()[step].cpu().numpy().ravel(), 100)
                    heights = np.repeat(n, 2)
                    hist[0].xy[1:-1, 1] = heights  # update heights of all but first and last entry
                title.set_text(f"Rollout Step {step+1} | Init {start_date} | Valid {start_date + pd.Timedelta(step+1, 'D')}")

            _, _, fc_hist_mean = ax.hist(forecast.mean(dim=0).squeeze()[0].cpu().numpy().ravel(), bins=100,
                                      histtype=u'step', log=True, label=f"UNet")
            _, _, fc_hist_trend = ax.hist(trend.squeeze()[0].cpu().numpy().ravel(), bins=100,
                                      histtype=u'step', log=True, label=f"Linear Trend Forecast")
            _, _, fc_hist_truth = ax.hist(truth.squeeze()[0].cpu().numpy().ravel(), bins=100,
                                         histtype=u'step', log=True, label=f"Truth")
            data = [forecast.mean(dim=0), trend, truth]
            hists = [fc_hist_mean, fc_hist_trend, fc_hist_truth]
            title = plt.suptitle(f"Rollout Step 1 | Init {start_date} | Valid {start_date + pd.Timedelta(1, 'D')}")
            plt.legend()
            plt.tight_layout()
            animation = FuncAnimation(fig, update_hist, range(self.num_rollout_steps), interval=1000 / fps)
            plt.close()
            animation.save("temp.mp4", fps=fps)
            wandb.log({"rollout_hist": wandb.Video("temp.mp4")})

            # create spectrum video
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            def update_spectrum(step):
                for d, line in zip(data, lines):
                    k, A = compute_spectrum2d(d.squeeze()[step].cpu().numpy())
                    line.set_ydata(A)
                title.set_text(f"Rollout Step {step+1} | Init {start_date} | Valid {start_date + pd.Timedelta(step+1, 'D')}")

            k, A = compute_spectrum2d(forecast.mean(dim=0).squeeze()[0].cpu().numpy())
            fc_line_mean, = ax.loglog(k, A, label=f"UNet", animated=True)
            k, A = compute_spectrum2d(trend.squeeze()[0].cpu().numpy())
            fc_line_trend, = ax.loglog(k, A, label=f"Linear Trend Forecast", animated=True)
            k, A = compute_spectrum2d(truth.squeeze()[0].cpu().numpy())
            fc_line_truth, = ax.loglog(k, A, label=f"Truth", animated=True)
            data = [forecast.mean(dim=0), trend, truth]
            lines = [fc_line_mean, fc_line_trend, fc_line_truth]
            title = plt.suptitle(f"Rollout Step 1 | Init {start_date} | Valid {start_date + pd.Timedelta(1, 'D')}")
            plt.xlabel("Wavenumber (Spatial Frequency)")
            plt.ylabel("Signal Power")
            plt.legend()
            plt.tight_layout()
            animation = FuncAnimation(fig, update_spectrum, range(self.num_rollout_steps), interval=1000 / fps)
            plt.close()
            animation.save("temp.mp4", fps=fps)
            wandb.log({"rollout_spectrum": wandb.Video("temp.mp4")})

            # log rollout metrics using ensemble mean
            rollout_preds = forecast.mean(dim=0).unsqueeze(0)
            rollout_target = truth.unsqueeze(0)
            rollout_sw = sample_weight[[0]].repeat(1, self.num_rollout_steps, 1, 1)
            self.val_rollout_metrics.update(rollout_preds, rollout_target, rollout_sw)
            self.log_dict(self.val_rollout_metrics.compute())
            self.val_rollout_metrics.reset()
    

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.val_metrics.reset()
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # torch.save(self.model, f"results/lightning-diffusion-epoch-{self.current_epoch}-{dt}.pth")

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.test_metrics.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return [opt], []  # add schedulers to second list if desired


def train_icenet(args):
    """
    Train IceNet using the arguments specified in the `args` namespace.
    :param args: Namespace of configuration parameters
    """
    # init
    pl.seed_everything(args.seed)
    
    # configure datasets and dataloaders
    if args.overfit:
        train_dataset = OverfitDatasetAMSR("/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera", "20190826", "20190827", args.overfit_replicates)
        val_dataset = OverfitDatasetAMSR("/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera", "20190826", "20190827", args.batch_size)  # one batch for validation
    else:
        train_dataset = DatasetAMSR("/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera", "20130103", "20191231")
        val_dataset = DatasetAMSR("/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera", "20200103", "20201231")
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  persistent_workers=True if args.num_workers > 0 else False, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                persistent_workers=True if args.num_workers > 0 else False, shuffle=True)

    # get number of channels
    x, y, sw, date = train_dataset[0]
    in_channels = x.shape[0]
    out_channels = y.shape[0]
    resolution = x.shape[-1]  # x is (b, c, h, w)

    # configure PyTorch Lightning module
    if args.model == "diffusion":
        if args.scheduler == "ddpm":
            noise_scheduler = DDPMScheduler(num_train_timesteps=args.scheduler_timesteps, prediction_type=args.prediction_type, rescale_betas_zero_snr=True, clip_sample_range=1.0)  # nb: change to prevent histogram collapse at -1, +1?
        elif args.scheduler == "ddim":
            noise_scheduler = DDIMScheduler(num_train_timesteps=args.scheduler_timesteps, prediction_type=args.prediction_type, rescale_betas_zero_snr=True, timestep_spacing="trailing")
        elif args.scheduler == "heun":
            noise_scheduler = HeunDiscreteScheduler(num_train_timesteps=args.scheduler_timesteps, prediction_type=args.prediction_type)
        else:
            raise NotImplementedError(f"Scheduler selection invalid: {args.scheduler} is not a supported scheduler type")
        lit_module = LitDiffusion(in_channels=in_channels,
                                  out_channels=out_channels,
                                  learning_rate=args.learning_rate,
                                  num_train_timesteps=args.scheduler_timesteps,
                                  num_inference_timesteps=args.scheduler_timesteps,
                                  noise_scheduler=noise_scheduler,
                                  prediction_type=args.prediction_type,
                                  forecast_residuals=args.forecast_residuals,
                                  resolution=resolution)
    elif args.model == "unet":
        lit_module = LitUNet(in_channels=in_channels,
                             out_channels=out_channels,
                             learning_rate=args.learning_rate,
                             forecast_residuals=args.forecast_residuals,
                             resolution=resolution)
    else:
        raise NotImplementedError(f"Model selection invalid: {args.model} is not a supported model type")
            
    # set up wandb logging
    wandb.init(project="icenet-diffusion")
    if args.name != "default":
        wandb.run.name = args.name
    wandb_logger = pl.loggers.WandbLogger(project="icenet-diffusion")
    wandb_logger.experiment.config.update(args)

    # comment/uncomment the following line to track gradients
    # note that wandb cannot parallelise across multiple gpus when tracking gradients
    # wandb_logger.watch(model, log="all", log_freq=10)
    best_checkpoint_callback = ModelCheckpoint(dirpath="results/", 
                                            filename="amsr-rothera-" +
                                                f"{args.dataloader_config}-" +
                                                f"{args.model}-{args.scheduler}{args.scheduler_timesteps}-" +
                                                f"{args.prediction_type}-" +
                                                f"res-{args.forecast_residuals}-" +
                                                "{epoch}-{val_mse:.4f}",
                                            save_last=False,
                                            monitor="val_mse")
    last_checkpoint_callback = ModelCheckpoint(dirpath="results/", 
                                          filename="amsr-rothera-" +
                                            f"{args.dataloader_config}-" +
                                            f"{args.model}-{args.scheduler}{args.scheduler_timesteps}-" +
                                            f"{args.prediction_type}-" +
                                            f"res-{args.forecast_residuals}-" +
                                             "{epoch}-{val_mse:.4f}",
                                          save_last=False)
    
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
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=1,
        callbacks=[
            best_checkpoint_callback,
            last_checkpoint_callback
        ]
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
    parser.add_argument("--model", default="diffusion", type=str, choices=["diffusion", "unet"],
                        help="Choice of model architecture", required=False)
    
    # model configurations applicable to all
    parser.add_argument("--dataloader_config", default="dataset_config.ice_only_lag_two.json", type=str,
                        help="Filename of dataloader_config.json file")
    parser.add_argument("--forecast_residuals", default=False, type=eval, help="Whether to predict residuals")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    # model configurations applicable to diffusion only
    parser.add_argument("--scheduler", default="ddpm", type=str, choices=["ddpm", "ddim", "heun"], help="Noise scheduler strategy")
    parser.add_argument("--scheduler_timesteps", default=100, type=int, help="Number of noising timesteps")
    parser.add_argument("--prediction_type", default="epsilon", type=str, choices=["epsilon", "v_prediction", "sample"], help="Scheduler prediction type")

    # hardware configurations applicable to all
    parser.add_argument("--accelerator", default="auto", type=str, help="PytorchLightning training accelerator")
    parser.add_argument("--devices", default=1, type=int, help="PytorchLightning number of devices to run on")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of workers in dataloader")
    parser.add_argument("--precision", default=32, type=int, choices=[32, 16], help="Precision for training")
    
    # debugging configurations applicable to all
    parser.add_argument("--overfit", default=False, type=eval, help="Whether to overfit on a subset for debugging")
    parser.add_argument("--overfit_replicates", default=10_000, type=str, help="Number of replicates on which to overfit if debugging")

    # logging configurations applicable to all
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

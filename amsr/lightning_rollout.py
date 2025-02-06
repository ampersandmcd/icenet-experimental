import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import sys
import pandas as pd
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import viridis, tab10


def eval_icenet(args):

    from lightning_train import DatasetAMSR, OverfitDatasetAMSR, LitDiffusion
    import lightning.pytorch as pl
    import wandb
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torchvision.ops.focal_loss import sigmoid_focal_loss
    from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, HeunDiscreteScheduler
    import torch.nn.functional as F
    import lightning.pytorch as pl
    from torchmetrics import MetricCollection
    from utils import IceNetDataSetPyTorch, IterableIceNetDataSetPyTorch

    model = LitDiffusion.load_from_checkpoint(args.path)
    start_date = pd.to_datetime(args.start_date, format="%Y%m%d")
    end_date = start_date + pd.Timedelta(args.days_to_rollout - 1, "days")
    date_range = pd.date_range(start_date, end_date)

    dataset = DatasetAMSR("/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera", args.start_date, end_date.strftime('%Y%m%d'))
    x, y, sample_weight, dates = dataset[0]
    x = x.to(model.device)
    y = y.to(model.device)
    sample_weight = sample_weight.to(model.device)
    samples = []
    
    path = f"/data/hpcdata/users/anddon76/icenet/icenet-experimental/amsr/results/{args.name}"
    os.makedirs(path, exist_ok=True)
    np.save(f"{path}/persist.npy", x[0, 0].cpu().numpy())  # save persistence forecast before we do anything

    x = x.repeat(args.ensemble_size, 1, 1, 1).to(model.device)  # copy same x conditions to num_samples size batch
    mask = torch.where(sample_weight[0] > 0, 1, 0).to(model.device)
    forecast = []

    # iterate over days
    for date in tqdm(date_range):

        # sample rollout
        noisy_y = torch.randn(args.ensemble_size, model.out_channels,
                            model.resolution, model.resolution).to(model.device) / 1000  # TODO RESCALE NOISE
        for t in model.noise_scheduler.timesteps:
            with torch.no_grad():
                pred = model.forward(noisy_y, t, x)
            noisy_y = model.noise_scheduler.step(pred, t, noisy_y).prev_sample
        
        if model.forecast_residuals:
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
        next_context = torch.from_numpy(next_sample[3:]).nan_to_num().to(torch.float).to(model.device)
        next_context = next_context.repeat(args.ensemble_size, 1, 1, 1)  # broadcast copies for ensemble

        # note sample = [ice_t, ice_t_1, ice_t_2, clim, land, cos, sin]
        # note x = [ice_t_1, ice_t_2, clim, land, cos, sin]            
        x = torch.cat((
            noisy_y,   # set ice_t -> ice_t_1 from prediction
            x[:, [0]],  # set ice_t_1 -> ice_t_2 from last predictor
            next_context, # climatology, land, and cos/sin embeddings for next day without
        ), dim=1)  # update to next timestep, t is first, t-1 is second
        x = x.to(dtype=torch.float32)
        
    # concatenate into forecast
    forecast = torch.cat(forecast, dim=1).to(model.device)

    # compile ground truth and climatology
    # note from dataset creation sample = np.array([ice_t, ice_t_1, ice_t_2, clim, land, cos, sin])
    truth, clim = [], []
    date = start_date
    while date <= end_date:
        f = f"/data/hpcdata/users/anddon76/icenet/icenet-experimental/processed/amsr_rothera/{date.strftime('%Y%m%d')}.npy"
        sample = np.load(f)
        truth.append(sample[0])  # ice_t is first element of sample
        clim.append(sample[3])  # clim is fourth element of sample
        date = date + pd.Timedelta(1, "day")
    truth = torch.from_numpy(np.array(truth)).nan_to_num().to(torch.float).to(model.device)
    clim = torch.from_numpy(np.array(clim)).nan_to_num().to(torch.float).to(model.device)

    # save results
    np.save(f"{path}/forecast.npy", forecast.cpu().numpy())
    np.save(f"{path}/truth.npy", truth.cpu().numpy())
    np.save(f"{path}/clim.npy", clim.cpu().numpy())
    np.save(f"{path}/mask.npy", mask.cpu().numpy())
    np.savetxt(f"{path}/dates.txt", date_range, fmt="%s")


def videos_icenet(args):

    from utils import compute_spectrum2d

    # load results
    path = f"/data/hpcdata/users/anddon76/icenet/icenet-experimental/amsr/results/{args.name}"
    forecast = np.load(f"{path}/forecast.npy")
    forecast_mean = forecast.mean(axis=0)
    forecast_std = forecast.std(axis=0)
    truth = np.load(f"{path}/truth.npy")
    clim = np.load(f"{path}/clim.npy")
    mask = np.load(f"{path}/mask.npy")
    dates = np.loadtxt(f"{path}/dates.txt", dtype=datetime)

    # video 1, rollout ensemble multiview
    fig, ax = plt.subplots(4, 8, figsize=(40, 21))
    ax = ax.ravel()
    frames = []
    fps = 3
    def rollout_ensemble_update(step):
        for member, f in enumerate(frames):
            f.set_data(forecast[member][step])
        title.set_text(f"Rollout Step {step+1} | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[step]}")
    for i in range(args.ensemble_size):
        f = ax[i].imshow(forecast[i][0], vmin=0, vmax=1, animated=True)
        frames.append(f)
    title = plt.suptitle(f"Rollout Step 1 | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[0]}")
    plt.tight_layout()
    animation = FuncAnimation(fig, rollout_ensemble_update, range(len(dates)), interval=1000 / fps)
    plt.close()
    animation.save(f"{path}/rollout_ensemble.mp4", fps=fps)

    # video 2, rollout summary statistics
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax = ax.ravel()
    fps = 3
    def rollout_summary_update(step):
        ftruth.set_data(truth[step])
        fmean.set_data(forecast_mean[step])
        fdiff.set_data(forecast_mean[step] - truth[step])
        fstd.set_data(forecast_std[step])
        title.set_text(f"Rollout Step {step+1} | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[step]}")
    ftruth = ax[0].imshow(truth[0], vmin=0, vmax=1, animated=True)
    ax[0].set_title("Truth")
    fmean = ax[1].imshow(forecast_mean[0], vmin=0, vmax=1, animated=True)
    ax[1].set_title(f"Ensemble Mean (n={args.ensemble_size})")
    fdiff = ax[2].imshow(forecast_mean[0] - truth[0], cmap="bwr", vmin=-1, vmax=1, animated=True)
    ax[2].set_title(f"Ensemble Mean (n={args.ensemble_size}) - Truth")
    fstd = ax[3].imshow(forecast_std[0], cmap="cividis", animated=True)
    ax[3].set_title(f"Ensemble Std (n={args.ensemble_size})")
    title = plt.suptitle(f"Rollout Step 1 | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[0]}")
    plt.tight_layout()
    animation = FuncAnimation(fig, rollout_summary_update, range(len(dates)), interval=1000 / fps)
    plt.close()
    animation.save(f"{path}/rollout_summary.mp4", fps=fps)

    # video 3, histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fps = 3
    def rollout_hist_update(step):
        # a bit hacky since we are using histtype="step"
        for d, hist in zip(data, hists):
            n, _ = np.histogram(d[step].ravel(), 100)
            heights = np.repeat(n, 2)
            hist[0].xy[1:-1, 1] = heights  # update heights of all but first and last entry
        title.set_text(f"Rollout Step {step+1} | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[step]}")
    data = []
    hists = []
    for i in range(args.ensemble_size):
        _, _, h = ax.hist(forecast[i].ravel().clip(0, 1), bins=100, histtype=u'step', log=True, color=viridis(i/args.ensemble_size))
        data.append(forecast[i])  # need single-level indexing
        hists.append(h)
    _, _, hist_mean = ax.hist(forecast_mean[0].ravel(), bins=100, histtype=u'step', log=True, label=f"Mean", color="magenta")
    _, _, hist_truth = ax.hist(truth[0].ravel(), bins=100, histtype=u'step', log=True, label=f"Truth", color="black")
    data = data + [forecast_mean, truth]
    hists = hists + [hist_mean, hist_truth]
    title = plt.suptitle(f"Rollout Step 1 | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[0]}")
    plt.legend()
    plt.tight_layout()
    animation = FuncAnimation(fig, rollout_hist_update, range(len(dates)), interval=1000 / fps)
    plt.close()
    animation.save(f"{path}/rollout_hist.mp4", fps=fps)

    # video 4, spectrum
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fps = 3
    def update_spectrum(step):
        for d, line in zip(data, lines):
            k, A = compute_spectrum2d(d[step])
            line.set_ydata(A)
        title.set_text(f"Rollout Step {step+1} | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[step]}")
    data = []
    lines = []
    for i in range(args.ensemble_size):
        k, A = compute_spectrum2d(forecast_mean[0])
        line, = ax.loglog(k, A, animated=True, color=viridis(i/args.ensemble_size))
        data.append(forecast[i])  # need single-level indexing
        lines.append(line)
    k, A = compute_spectrum2d(forecast_mean[0])
    line_mean, = ax.loglog(k, A, label=f"Mean", animated=True, color="magenta")
    k, A = compute_spectrum2d(truth[0])
    line_truth, = ax.loglog(k, A, label=f"Truth", animated=True, color="black")
    data = data + [forecast_mean, truth]
    lines = lines + [line_mean, line_truth]
    title = plt.suptitle(f"Rollout Step 1 | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[0]}")
    plt.xlabel("Wavenumber (Spatial Frequency)")
    plt.ylabel("Signal Power")
    plt.legend()
    plt.tight_layout()
    animation = FuncAnimation(fig, update_spectrum, range(len(dates)), interval=1000 / fps)
    plt.close()
    animation.save(f"{path}/rollout_spectrum.mp4", fps=fps)

    # video 5, edges
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fps = 3
    def rollout_edge_update(step):
        for artist in ax[0].collections:
            artist.remove()
        for member in range(32):
            arr = forecast[member, step]
            X, Y = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
            ax[0].contour(X, Y, arr, [0.15], colors=[viridis(member/32)])
        # landmask
        ax[0].contour(X, Y, mask, [0.15], colors=["black"])
        # groundtruth
        ax[0].contour(X, Y, truth[step], [0.15], colors=["magenta"])
        im.set_data(np.where(mask, truth[step], np.nan))
        title.set_text(f"Rollout Step {step+1} | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[step]}")
    # forecasts
    for member in range(32):
        arr = forecast[member, 0]
        X, Y = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
        ax[0].contour(X, Y, arr, [0.15], colors=[viridis(member/32)])
    # landmask
    ax[0].contour(X, Y, mask, [0.15], colors=["black"])
    # groundtruth
    ax[0].contour(X, Y, truth[0], [0.15], colors=["magenta"])
    ax[0].set_title("Forecast and True Edge")
    ax[0].invert_yaxis()
    ax[0].plot(0, 0, color="magenta", label="True Edge")  # hack to get plot label
    ax[0].plot(0, 0, color="black", label="Land Mask")  # hack to get plot label
    ax[0].legend(loc="lower left")
    # plot true ice for reference
    im = ax[1].imshow(np.where(mask, truth[0], np.nan))
    ax[1].set_title("True SIC")
    title = plt.suptitle(f"Rollout Step 1 | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Valid {dates[0]}")
    plt.tight_layout()
    animation = FuncAnimation(fig, rollout_edge_update, range(len(dates)), interval=1000 / fps)
    plt.close()
    animation.save(f"{path}/rollout_edge.mp4", fps=fps)


def plots_icenet(args):

    # load results
    path = f"/data/hpcdata/users/anddon76/icenet/icenet-experimental/amsr/results/{args.name}"
    forecast = np.load(f"{path}/forecast.npy")
    forecast_mean = forecast.mean(axis=0)
    forecast_std = forecast.std(axis=0)
    truth = np.load(f"{path}/truth.npy")
    clim = np.load(f"{path}/clim.npy")
    mask = np.load(f"{path}/mask.npy")
    persist = np.load(f"{path}/persist.npy")
    dates = np.loadtxt(f"{path}/dates.txt", dtype=datetime)

    # plot 1, rmse (recall shape is [ensemble, date, h, w])
    rmses = []
    for member in range(args.ensemble_size):
        rmse = np.sqrt(np.sum((forecast[member] - truth)**2, axis=(-1, -2)) / np.sum(mask))
        rmses.append(rmse)
    rmse_mean = np.sqrt(np.sum((forecast_mean - truth)**2, axis=(-1, -2)) / np.sum(mask))
    rmse_clim = np.sqrt(np.sum((clim - truth)**2, axis=(-1, -2)) / np.sum(mask))
    rmse_persist = np.sqrt(np.sum((persist - truth)**2, axis=(-1, -2)) / np.sum(mask))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for member in range(args.ensemble_size):
        rmse = rmses[member]
        ax.plot(range(1, len(rmse)+1), rmse, color=viridis(member/args.ensemble_size))
    ax.plot(range(1, len(rmse_mean)+1), rmse_mean, color="magenta", label="Mean", linestyle=":")
    ax.plot(range(1, len(rmse_clim)+1), rmse_clim, color="red", label="Climatology", linestyle=":")
    ax.plot(range(1, len(rmse_persist)+1), rmse_persist, color="orange", label="Persistence", linestyle=":")
    plt.title(f"Rollout RMSE | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Through {dates[-1]}")
    plt.legend()
    plt.xlabel("Forecast Leadtime")
    plt.ylabel("Forecast RMSE")
    plt.tight_layout()
    plt.savefig(f"{path}/rollout_rmse.png")
    plt.close()

    # plot 2, binary accuracy (recall shape is [ensemble, date, h, w])
    accs = []
    bin_truth = np.where(truth > 0.15, 1, 0)
    for member in range(args.ensemble_size):
        acc = np.sum(np.logical_and(np.where(forecast[member] > 0.15, 1, 0) == bin_truth, mask), axis=(-1, -2)) / np.sum(mask)
        accs.append(acc.squeeze())
    acc_mean = (np.sum(np.logical_and(np.where(forecast_mean > 0.15, 1, 0) == bin_truth, mask), axis=(-1, -2)) / np.sum(mask)).squeeze()
    acc_clim = (np.sum(np.logical_and(np.where(clim > 0.15, 1, 0) == bin_truth, mask), axis=(-1, -2)) / np.sum(mask)).squeeze()
    acc_persist = (np.sum(np.logical_and(np.where(persist > 0.15, 1, 0) == bin_truth, mask), axis=(-1, -2)) / np.sum(mask)).squeeze()
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for member in range(args.ensemble_size):
        acc = accs[member]
        ax.plot(range(1, len(acc)+1), acc, color=viridis(member/args.ensemble_size))
    ax.plot(range(1, len(acc_mean)+1), acc_mean, color="magenta", label="Mean", linestyle=":")
    ax.plot(range(1, len(acc_clim)+1), acc_clim, color="red", label="Climatology", linestyle=":")
    ax.plot(range(1, len(acc_persist)+1), acc_persist, color="orange", label="Persistence", linestyle=":")
    plt.title(f"Rollout Binary Accuracy | Init {pd.to_datetime(dates[0]) - pd.Timedelta(1, 'D')} | Through {dates[-1]}")
    plt.legend()
    plt.xlabel("Forecast Leadtime")
    plt.ylabel("Forecast Binary Accuracy")
    plt.tight_layout()
    plt.savefig(f"{path}/rollout_acc.png")
    plt.close()

    # plot 3, crps over time

    # plot 4, spread skill ratio

    # plot 5, rank histogram


if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser(description="Eval IceNet")
    parser.add_argument("--path", default="/data/hpcdata/users/anddon76/icenet/icenet-experimental/amsr/results/amsr-rothera-dataset_config.ice_only_lag_two.json-diffusion-ddpm1000-v_prediction-res-False-epoch=44-val_mse=0.0104.ckpt", 
                        type=str, help="Path to model checkpoint")
    parser.add_argument("--name", default="jan2021_30day_32member", 
                        type=str, help="Unique name under which to save rollout result")
    parser.add_argument("--ensemble_size", default=32, type=int,
                         help="Number of ensemble members", required=False)
    parser.add_argument("--days_to_rollout", default=30, type=int,
                         help="Number of days to rollout", required=False)
    parser.add_argument("--start_date", default=20210101, type=int,
                         help="Start date of forecast as integer", required=False)
    
    # let's go
    args = parser.parse_args()
    eval_icenet(args)
    videos_icenet(args)
    plots_icenet(args)

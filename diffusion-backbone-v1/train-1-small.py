import os
import torch
from diffusers import AutoencoderKL, UNet2DModel, PNDMScheduler, DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.utils import make_image_grid, load_image
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm.auto import tqdm
from diffusers.utils import make_image_grid
import matplotlib.pyplot as plt
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_piecewise_constant_schedule
from utils import IceNetDataSetPyTorch, IterableIceNetDataSetPyTorch, BinaryAccuracy, SIEError, SIAError, RMSError, MAError
from torchmetrics import MetricCollection
from datetime import datetime
import numpy as np
from icenet.data.loaders import IceNetDataLoaderFactory
import wandb

# don't do this permanently but for now the warnings are annoying
import warnings
warnings.filterwarnings("ignore")

@dataclass
class TrainingConfig:
    image_size = 432  # the generated image resolution, must match training dataset size
    in_channels = 10
    mid_channels = 128
    out_channels = 1
    train_batch_size = 4
    eval_batch_size = 4
    num_epochs = 10000
    gradient_accumulation_steps = 1
    learning_rate = 2e-6
    lr_warmup_steps = 10
    validate_epochs = 100
    save_model_epochs = 10000
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results/train-1"  # the model name locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0
    num_train_timesteps = 4  # this is small but makes inference fast, try increasing later
    use_mask = False
    dataset_config = "dataset_config.single_day.json"
    loader_config = "loader.single_day.json"
    cross_attention_dim = 512
    reconstruction_factor = 1
    coupled_factor = 1
    num_workers = 1


def validate(config, epoch, unet, scheduler, val_dl, accelerator):

    with torch.autocast(device_type="cuda"):

        metrics = {
            "val_rmse": RMSError(leadtimes_to_evaluate=list(range(config.out_channels))).to("cuda"),
            "val_mae": MAError(leadtimes_to_evaluate=list(range(config.out_channels))).to("cuda"),
            "val_accuracy": BinaryAccuracy(leadtimes_to_evaluate=list(range(config.out_channels))).to("cuda"),
            "val_sieerror": SIEError(leadtimes_to_evaluate=list(range(config.out_channels))).to("cuda"),
            "val_siaerror": SIAError(leadtimes_to_evaluate=list(range(config.out_channels))).to("cuda")
        }
        metrics = MetricCollection(metrics)

        # define these outside of batch loop for use in plotting later
        x, y, sw = None, None, None

        progress_bar = tqdm(total=len(val_dl), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Validating {epoch}")

        for step, batch in enumerate(val_dl):

            if step >= 1:
                break  # TODO: change this later, just for speed for now

            # set eval mode
            unet.eval()

            with torch.no_grad():

                # sample images from batch
                x, y, sw = batch

                # subtract previous ice state since network to predicts residual
                previous_ice = x[:, [2], :, :]
                targets = y - previous_ice

                # sample noise to add to the images
                noise = torch.randn(targets.shape, device=targets.device)
                sample = noise

                # set timesteps and denoise
                scheduler.set_timesteps(config.num_train_timesteps)
                mat_samples = []

                for i, t in enumerate(tqdm(scheduler.timesteps)):

                    # scale latents
                    sample_model_input = scheduler.scale_model_input(sample, timestep=t)

                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = unet(torch.cat((sample_model_input, x), dim=1), t).sample
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    sample = scheduler.step(noise_pred, t, sample).prev_sample

                    # save some for plotting
                    if i % 4 == 0:
                        mat_samples.append(sample.squeeze().detach().cpu().numpy())

                # add previous ice state since network is trained to predict residual
                outputs = sample + previous_ice

                # mask if using mask
                if config.use_mask:
                    outputs = outputs * torch.where(sw > 0.0, 1.0, 0.0)

                # clip outputs to be in range (0, 1)
                outputs = torch.clip(outputs, 0, 1)

                # update metrics
                metrics.update(outputs, y, sw)

                progress_bar.update(1)

        # compute metrics over entire validation set
        accelerator.log(metrics.compute())  # epoch-level metrics
        metrics.reset()

        # make plot of selected inputs/outputs
        fig, ax = plt.subplots(3, 5, figsize=(25, 15))
        mat_in = x[0].squeeze().detach().cpu().numpy()
        mat_out = outputs[0].squeeze().detach().cpu().numpy()
        mat_residual = sample[0].squeeze().detach().cpu().numpy()
        mat_true = y[0].squeeze().detach().cpu().numpy()

        # first row
        ax[0][0].imshow(mat_in[2])
        ax[0][0].set_title("Initial")
        ax[0][1].imshow(mat_out)
        ax[0][1].set_title("Output @ t=10")
        ax[0][2].imshow(mat_true)
        ax[0][2].set_title("Truth @ t=10")
        ax[0][3].imshow(mat_out - mat_true, cmap="bwr", vmin=-1, vmax=1)
        ax[0][3].set_title(f"Forecast - Truth")
        ax[0][4].imshow(mat_out - mat_in[2], cmap="bwr", vmin=-1, vmax=1)
        ax[0][4].set_title(f"Forecast - Initial")
        
        # second row
        ax[1][0].imshow(mat_in[2])
        ax[1][0].set_title("Initial")
        ax[1][1].imshow(mat_residual)
        ax[1][1].set_title("Residual Output @ t=10")
        ax[1][2].imshow(mat_true - previous_ice[0].squeeze().detach().cpu().numpy())
        ax[1][2].set_title("Residual Truth @ t=10")
        ax[1][3].axis("off")
        ax[1][4].axis("off")

        # third row
        for i in range(5):
            if i < len(mat_samples):
                ax[2][i].imshow(mat_samples[i])
                ax[2][i].set_title(f"Denoising Step {i*4}")
            else:
                ax[2][i].axis("off")

        plt.tight_layout()

        # log outputs
        accelerator.log({"Sample": plt})

        # save the outputs
        sample_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"{sample_dir}/{timestring}_{epoch:04d}.png")
        plt.close()


def train():

    with torch.autocast(device_type="cuda"):

        # load config hyperparameters
        config = TrainingConfig()

        # set up models
        unet = UNet2DModel(
            in_channels = config.in_channels,  # from predictors and noise
            out_channels = config.out_channels,  # to ice
            block_out_channels = (8, 16, 32, 64, 128),
            down_block_types=tuple(["DownBlock2D"] * 5),
            up_block_types=tuple(["UpBlock2D"] * 5),
            norm_num_groups=1
        )
        print(f"UNet Parameters: {sum([p.numel() for p in unet.parameters()])}")

        # init scheduler
        # use DPM++ 2M Karras, said to be "one of the best" samplers at the moment
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=config.num_train_timesteps, use_karras_sigmas=True)

        # set up dataset
        train_ds = IceNetDataSetPyTorch(config.dataset_config, "train", batch_size=config.train_batch_size, shuffling=False)
        val_ds = IceNetDataSetPyTorch(config.dataset_config, "val", batch_size=config.train_batch_size, shuffling=False)
        test_ds = IceNetDataSetPyTorch(config.dataset_config, "test", batch_size=config.train_batch_size, shuffling=False)

        # configure dataloaders
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.eval_batch_size, shuffle=False)

        # configure optimizers
        optimizer = torch.optim.AdamW(
                params=unet.parameters(),
                lr=config.learning_rate)
        scheduler.set_timesteps(config.num_train_timesteps)

        # initialize accelerator and wandb logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb",
            project_dir=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            if config.push_to_hub:
                repo_id = create_repo(
                    repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
                ).repo_id
            accelerator.init_trackers(project_name="icenet-diffusion")
        # wandb.watch(unet, log="all")
        # wandb.watch(conditioner, log="all")

        # prepare objects for accelerator
        unet, optimizer, optimizer, train_dl, val_dl = accelerator.prepare(
            unet, optimizer, optimizer, train_dl, val_dl
        )

        # begin training loop
        global_step = 0
        for epoch in range(config.num_epochs):

            progress_bar = tqdm(total=len(train_dl), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dl):

                # set training mode
                unet.train()

                # unpack batch
                x, y, sw = batch

                # subtract previous ice state to train network to predict residual
                previous_ice = x[:, [2], :, :]
                targets = y - previous_ice

                # sample noise to add to the images
                noise = torch.randn(targets.shape, device=targets.device)

                # sample a random timestep for each image
                bs = targets.shape[0]
                timestep_indices = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps, 
                    (bs,),
                    dtype=torch.int64
                )
                scheduler.set_timesteps(config.num_train_timesteps)
                timesteps = scheduler.timesteps[timestep_indices].to(targets.device)

                # add noise to clean images in forward diffusion process
                noisy_images = scheduler.add_noise(targets, noise, timesteps)

                with accelerator.accumulate(unet):

                    # predict the noise residual
                    noise_pred = unet(torch.cat((noisy_images, x), dim=1), timesteps).sample

                    # compute mse loss on residual (training u-net)
                    loss = F.mse_loss(noise_pred, noise)

                    # sum and backprop
                    accelerator.backward(loss)

                    # step optimizers
                    optimizer.step()
                    optimizer.zero_grad()

                # update logs and trackers
                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "step": global_step
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                global_step += 1

            # optionally sample some demo forecassts with evaluate() and save the model
            if accelerator.is_main_process:

                if epoch % config.validate_epochs == 0 or epoch == config.num_epochs - 1:
                    validate(config, epoch, unet, scheduler, val_dl, accelerator)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if config.push_to_hub:
                        upload_folder(
                            repo_id=repo_id,
                            folder_path=config.output_dir,
                            commit_message=f"Epoch {epoch}",
                            ignore_patterns=["step_*", "epoch_*"],
                        )
                    else:
                        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        torch.save(unet, Path(config.output_dir, dt + "_unet.pth"))


if __name__ == "__main__":
    train()

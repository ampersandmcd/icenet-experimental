import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DPMSolverMultistepScheduler
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

# don't do this permanently but for now the warnings are annoying
import warnings
warnings.filterwarnings("ignore")

@dataclass
class TrainingConfig:
    image_size = 432  # the generated image resolution, must match training dataset size
    in_channels = 26
    mid_channels = 128
    out_channels = 93
    train_batch_size = 1
    eval_batch_size = 1
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 100
    validate_epochs = 1
    save_model_epochs = 1
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results/end2endtrain93"  # the model name locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0
    num_train_timesteps = 50  # this is small but makes inference fast, try increasing later
    use_mask = True
    dataset_config = "dataset_config.exp23_south.json"
    loader_config = "loader.exp23_south.json"
    cross_attention_dim = 512
    reconstruction_factor = 1
    coupled_factor = 1
    num_workers = 16


def validate(config, epoch, unet, condition_ae, scheduler, val_dl, accelerator):

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

            if step >= 4:
                break  # TODO: change this later, just for speed for now

            # set eval mode
            unet.eval()
            condition_ae.eval()

            with torch.no_grad():

                # sample images from batch
                x, y, sw = batch
                # x = F.interpolate(x, size=(54, 54))
                # y = F.interpolate(y, size=(54, 54))
                # sw = F.interpolate(sw, size=(54, 54))

                # encode conditioning variables
                encoder_hidden_states = condition_ae.encode(x).latent_dist.mode()  # output is channels x 3 x 3
                encoder_hidden_states = F.avg_pool2d(encoder_hidden_states, kernel_size=(3, 3)).squeeze(dim=-1)  # pool to channels
                encoder_hidden_states = encoder_hidden_states.transpose(-1, 1)  # put seq length dim in position 1

                # subtract previous ice state since network to predicts residual
                previous_ice = x[:, [2], :, :]
                targets = y - previous_ice

                # sample noise to add to the images
                noise = torch.randn(targets.shape, device=targets.device)

                # set timesteps and denoise
                scheduler.set_timesteps(config.num_train_timesteps)
                sample = noise  # init from noise

                for t in tqdm(scheduler.timesteps):

                    # scale latents
                    sample_model_input = scheduler.scale_model_input(sample, timestep=t)

                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = unet(sample_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    sample = scheduler.step(noise_pred, t, sample).prev_sample

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
        fig, ax = plt.subplots(4, 5, figsize=(25, 20))
        mat_in = x[0].squeeze().detach().cpu().numpy()
        mat_out = outputs[0].squeeze().detach().cpu().numpy()
        mat_true = y[0].squeeze().detach().cpu().numpy()
        ax[0][0].imshow(mat_in[2])
        ax[0][0].set_title("Initial")
        vmax = 0
        for j, timestep in enumerate([0, 30, 60, 90]):
            amax = np.amax(np.abs(mat_out[timestep] - mat_true[timestep]))
            amax = np.amax(np.abs(mat_out[timestep] - mat_in[2]))
            vmax = max(amax, vmax)
        for j, timestep in enumerate([0, 30, 60, 90]):
            ax[0][j+1].imshow(mat_true[timestep], vmin=0, vmax=1)
            ax[0][j+1].set_title(f"True @ t={timestep}")
            mappable_vds = ax[1][j+1].imshow(mat_out[timestep], vmin=0, vmax=1)
            ax[1][j+1].set_title(f"Forecast @ t={timestep}")
            ax[2][j+1].imshow(mat_out[timestep] - mat_true[timestep], cmap="bwr", vmin=-vmax, vmax=vmax)
            ax[2][j+1].set_title(f"Forecast - Truth @ t={timestep}")
            mappable_bwr = ax[3][j+1].imshow(mat_out[timestep] - mat_in[2], cmap="bwr", vmin=-vmax, vmax=vmax)
            ax[3][j+1].set_title(f"Forecast - Initial @ t={timestep}")
        plt.colorbar(mappable_vds, cax=ax[1][0])
        plt.colorbar(mappable_bwr, cax=ax[2][0])
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
        unet = UNet2DConditionModel(
            in_channels = config.out_channels,  # from pure noise
            out_channels = config.out_channels,
            block_out_channels = (93, 93, 93, 93),  # (64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            cross_attention_dim = config.cross_attention_dim,
            norm_num_groups=1
        )
        conditioner = AutoencoderKL(
            in_channels = config.in_channels,
            out_channels = config.in_channels,
            down_block_types = tuple(["DownEncoderBlock2D"]*8),
            up_block_types = tuple(["UpDecoderBlock2D"]*8),
            block_out_channels = (config.in_channels, config.in_channels, config.in_channels, 32, 64, 128, 256, 512),
            norm_num_groups=1,
            latent_channels=config.cross_attention_dim
        )

        # init scheduler
        # use DPM++ 2M Karras, said to be "one of the best" samplers at the moment
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=config.num_train_timesteps, use_karras_sigmas=True)

        # set up dataset
        train_ds = IterableIceNetDataSetPyTorch(config.dataset_config, "train", batch_size=config.train_batch_size, shuffling=False)
        val_ds = IterableIceNetDataSetPyTorch(config.dataset_config, "val", batch_size=config.eval_batch_size, shuffling=False)
        test_ds = IterableIceNetDataSetPyTorch(config.dataset_config, "test", batch_size=config.eval_batch_size, shuffling=False)

        # configure dataloaders
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.eval_batch_size, shuffle=False)
        # implementation = "dask"
        # loader_config = config.loader_config
        # dataset_name = "exp23_south"
        # lag = 1
        # train_dl = IceNetDataLoaderFactory().create_data_loader(
        #     implementation,
        #     loader_config,
        #     dataset_name,
        #     lag,
        #     n_forecast_days=93,
        #     north=False,
        #     south=True,
        #     output_batch_size=config.train_batch_size,
        #     generate_workers=8
        # )

        # configure optimizers
        optimizer = torch.optim.AdamW(
                params=list(unet.parameters()) + list(conditioner.encoder.parameters()),
                lr=config.learning_rate)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
        )
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

        # prepare objects for accelerator
        unet, conditioner, optimizer, lr_scheduler, train_dl, val_dl = accelerator.prepare(
            unet, conditioner, optimizer, lr_scheduler, train_dl, val_dl
        )

        # begin training loop
        global_step = 0
        for epoch in range(config.num_epochs):

            # validate on first epoch
            if epoch == 0:  # TODO: change to val_dl later
                validate(config, epoch, unet, conditioner, scheduler, val_dl, accelerator)

            progress_bar = tqdm(total=len(train_dl), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dl):

                # set training mode
                unet.train()
                conditioner.train()

                # unpack batch
                x, y, sw = batch
                # x = F.interpolate(x, size=(54, 54))
                # y = F.interpolate(y, size=(54, 54))
                # sw = F.interpolate(sw, size=(54, 54))


                # encode conditioning variables
                conditioner_hidden_states = conditioner.encode(x).latent_dist.mode()  # output is channels x 3 x 3
                conditioner_hidden_states = F.avg_pool2d(conditioner_hidden_states, kernel_size=(3, 3)).squeeze(dim=-1)  # pool to channels
                conditioner_hidden_states = conditioner_hidden_states.transpose(-1, 1)  # put seq length dim in position 1

                # subtract previous ice state to train network to predict residual
                previous_ice = x[:, [2], :, :]
                targets = y - previous_ice

                # sample noise to add to the images
                noise = torch.randn(targets.shape, device=targets.device)

                # sample a random timestep for each image
                bs = targets.shape[0]
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps, 
                    (bs,),
                    device=targets.device,
                    dtype=torch.int64
                )

                # add noise to clean images in forward diffusion process
                scheduler.set_timesteps(config.num_train_timesteps)
                noisy_images = scheduler.add_noise(targets, noise, timesteps)

                with accelerator.accumulate(unet, conditioner):

                    # predict the noise residual
                    noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=conditioner_hidden_states).sample

                    # compute mse loss on residual (training u-net)
                    denoising_loss = F.mse_loss(noise_pred, noise)

                    # sum and backprop
                    loss = denoising_loss 
                    accelerator.backward(loss)

                    # step optimizers
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    accelerator.clip_grad_norm_(conditioner.encoder.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # update logs and trackers
                progress_bar.update(1)
                logs = {
                    "denoising_loss": denoising_loss.detach().item(),
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # optionally sample some demo forecassts with evaluate() and save the model
            if accelerator.is_main_process:

                if epoch % config.validate_epochs == 0 or epoch == config.num_epochs - 1:
                    validate(config, epoch, unet, conditioner, scheduler, val_dl, accelerator)

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
                        torch.save(conditioner, Path(config.output_dir, dt + "_condition_ae.pth"))


if __name__ == "__main__":
    train()
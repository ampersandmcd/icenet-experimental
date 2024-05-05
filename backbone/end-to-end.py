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
from utils import IceNetDataSetPyTorch
from datetime import datetime


@dataclass
class TrainingConfig:
    image_size = 432  # the generated image resolution, must match training dataset size
    latent_size = 54
    latent_channels = 1
    in_channels = 9
    mid_channels = 128
    out_channels = 1
    train_batch_size = 1
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 100
    save_image_epochs = 100
    save_model_epochs = 1000
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results/end2end"  # the model name locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0
    num_train_timesteps = 50  # this is small but makes inference fast, try increasing later
    use_mask = False
    dataset_config = "dataset_config.single_day.json"
    cross_attention_dim = 512
    rec_factor = 1


def evaluate(config, epoch, unet, condition_ae, ice_ae, scheduler, train_dl, accelerator):

    with torch.autocast(device_type="cuda"):

        # sample images from batch
        batch = next(iter(train_dl))
        x, y, sw = batch

        # encode conditioning variables
        encoder_hidden_states = condition_ae.encode(x).latent_dist.mode()  # output is channels x 3 x 3
        encoder_hidden_states = F.avg_pool2d(encoder_hidden_states, kernel_size=(3, 3)).squeeze(dim=-1)  # pool to channels
        encoder_hidden_states = encoder_hidden_states.transpose(-1, 1)  # put seq length dim in position 1

        # subtract previous ice state to train network to predict residual
        previous_ice = x[:, [2], :, :]
        targets = y - previous_ice
        target_latents = ice_ae.encode(targets).latent_dist.mode()  # map ice residuals to lower dim

        # generate two samples from different noise
        n_samples = 3
        samples, noises = [], []
        for _ in range(n_samples):

            # sample noise to add to the images
            noise = torch.randn(target_latents.shape, device=target_latents.device)

            # set timesteps and denoise
            scheduler.set_timesteps(config.num_train_timesteps - 1)
            latents = noise  # init from noise

            for t in tqdm(scheduler.timesteps):

                # scale latents
                latent_model_input = scheduler.scale_model_input(latents, timestep=t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # decode latent prediction
            outputs = ice_ae.decode(latents).sample

            # add previous ice state since network is trained to predict residual
            outputs = outputs + previous_ice

            # save to list of samples
            noises.append(noise)
            samples.append(outputs)

    # make plot of input/output
    fig, ax = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    for i in range(n_samples):
        mat_in = noises[i][0].squeeze().detach().cpu().numpy()
        mat_prev_ice = previous_ice.squeeze().detach().cpu().numpy()
        mat_out = samples[i][0].squeeze().detach().cpu().numpy()  # take first channel of output, this is sea ice
        mat_true = y[0].squeeze().detach().cpu().numpy()
        ax[i][0].imshow(mat_in)
        ax[i][0].set_title(f"Noise")
        ax[i][1].imshow(mat_prev_ice)
        ax[i][1].set_title(f"Previous Ice")
        ax[i][2].imshow(mat_out)
        ax[i][2].set_title(f"Forecast")
        ax[i][3].imshow(mat_true)
        ax[i][3].set_title(f"Truth")
        ax[i][4].imshow(mat_true - mat_out, cmap="bwr")
        ax[i][4].set_title(f"Truth - Forecast")
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

    # load config hyperparameters
    config = TrainingConfig()

    # set up models
    unet = UNet2DConditionModel(
        in_channels = 1,  # from pure noise
        out_channels = config.latent_channels,
        block_out_channels = (64, 128, 256, 512),
        cross_attention_dim = config.cross_attention_dim,
        norm_num_groups=1
    )
    ice_ae = AutoencoderKL(
        in_channels = config.latent_channels,
        out_channels = config.out_channels,
        down_block_types = tuple(["DownEncoderBlock2D"]*4),
        up_block_types = tuple(["UpDecoderBlock2D"]*4),
        block_out_channels = (8, 16, 32, 64),
        norm_num_groups=1,
        latent_channels=config.latent_channels  # maybe try changing this later to allow deeper representation
    )
    condition_ae = AutoencoderKL(
        in_channels = config.in_channels,
        out_channels = config.in_channels,
        down_block_types = tuple(["DownEncoderBlock2D"]*8),
        up_block_types = tuple(["UpDecoderBlock2D"]*8),
        block_out_channels = (config.in_channels, config.in_channels, 16, 32, 64, 128, 256, 512),
        norm_num_groups=1,
        latent_channels=config.cross_attention_dim
    )

    # init scheduler
    # use DPM++ 2M Karras, said to be "one of the best" samplers at the moment
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=config.num_train_timesteps, use_karras_sigmas=True)

    # set up dataset
    train_ds = IceNetDataSetPyTorch(config.dataset_config, "train", batch_size=config.train_batch_size, shuffling=False)
    val_ds = IceNetDataSetPyTorch(config.dataset_config, "val", batch_size=config.train_batch_size, shuffling=False)
    test_ds = IceNetDataSetPyTorch(config.dataset_config, "test", batch_size=config.train_batch_size, shuffling=False)

    # configure dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=False)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config.eval_batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.eval_batch_size, shuffle=False)

    # configure optimizers
    optimizer = torch.optim.AdamW(
            params=list(unet.parameters()) + list(condition_ae.encoder.parameters()) + list(ice_ae.parameters()),
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
    unet, condition_ae, ice_ae, optimizer, lr_scheduler, train_dl = accelerator.prepare(
        unet, condition_ae, ice_ae, optimizer, lr_scheduler, train_dl
    )

    # begin training loop
    global_step = 0
    for epoch in range(config.num_epochs):

        progress_bar = tqdm(total=len(train_dl), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dl):

            with torch.autocast(device_type="cuda"):

                # unpack batch
                x, y, sw = batch

                # encode conditioning variables
                encoder_hidden_states = condition_ae.encode(x).latent_dist.mode()  # output is channels x 3 x 3
                encoder_hidden_states = F.avg_pool2d(encoder_hidden_states, kernel_size=(3, 3)).squeeze(dim=-1)  # pool to channels
                encoder_hidden_states = encoder_hidden_states.transpose(-1, 1)  # put seq length dim in position 1

                # subtract previous ice state to train network to predict residual
                previous_ice = x[:, [2], :, :]
                targets = y - previous_ice
                target_latents = ice_ae.encode(targets).latent_dist.mode()  # map ice residuals to lower dim
                reconstructed_targets = ice_ae.decode(target_latents).sample  # reconstruct to train ae on residuals

                # sample noise to add to the images
                noise = torch.randn(target_latents.shape, device=target_latents.device)

                # sample a random timestep for each image
                bs = target_latents.shape[0]
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps, 
                    (bs,),
                    device=target_latents.device,
                    dtype=torch.int64
                )

                # add noise to clean images in forward diffusion process
                noisy_images = scheduler.add_noise(target_latents, noise, timesteps)

                with accelerator.accumulate(unet, condition_ae, ice_ae):

                    # predict the noise residual
                    noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                    # compute mse loss on residual
                    denoising_loss = F.mse_loss(noise_pred, noise)

                    # compute reconstruction loss from encoding-latent-decoding process on ice
                    reconstruction_loss = F.mse_loss(targets, reconstructed_targets)

                    # sum and backprop
                    loss = denoising_loss + config.rec_factor * reconstruction_loss
                    accelerator.backward(loss)

                    # step optimizers
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # update logs and trackers
                progress_bar.update(1)
                logs = {
                    "denoising_loss": denoising_loss.detach().item(),
                    "reconstruction_loss": reconstruction_loss.detach().item(),
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        # optionally sample some demo forecassts with evaluate() and save the model
        if accelerator.is_main_process:

            if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, unet, condition_ae, ice_ae, scheduler, train_dl, accelerator)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    torch.save(unet, Path(config.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pth"))


if __name__ == "__main__":
    train()

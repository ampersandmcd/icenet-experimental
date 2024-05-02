import os
import torch
from diffusers import UNet2DConditionModel, PNDMScheduler, DiffusionPipeline
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
    output_dir = "results/diffusion.img2img"  # the model name locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0
    num_train_timesteps = 200  # this is small but makes inference fast, try increasing later
    use_mask = False
    dataset_config = "dataset_config.single_day.json"
    cross_attention_dim = 1


def evaluate(config, epoch, unet, scheduler, train_dataloader):

    with torch.autocast(device_type="cuda"):

        # sample images from batch
        batch = next(iter(train_dataloader))
        x, y, sw = batch

        # DOWNSAMPLE
        x = F.interpolate(x, (54, 54))
        y = F.interpolate(y, (54, 54))
        sw = F.interpolate(sw, (54, 54))

        # create latents and pass latents through unet chain
        # latents = pipeline.vae.encode(x.to(torch.half).to("cuda")).latent_dist.mode()
        latents = x
        target_latents = y
        sw_latents = sw

        # sample noise to init denoising chain
        noise = torch.randn(target_latents.shape, device=target_latents.device)

        # concatenate predictors onto noise
        # latents = torch.cat((noise, latents), dim=1)  # init from noise and predictors
        latents = noise  # init from noise
        # latents = latents[:, [2], :, :]  # init from previous ice

        # set timesteps and denoise
        scheduler.set_timesteps(config.num_train_timesteps - 1)

        for t in tqdm(scheduler.timesteps):

            # scale latents
            latent_model_input = scheduler.scale_model_input(latents, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=torch.zeros((1, 1, config.cross_attention_dim), device="cuda")).sample
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # pass latents through decoder
        # outputs = pipeline.vae.decoder(latents)    
        outputs = latents
        if config.use_mask:
            outputs = latents * sw_latents
        else:
            outputs = latents

        # add previous ice state since network is trained to predict residual
        outputs = outputs + x[:, [2], :, :]

    # make plot of first input/output
    # fig, ax = plt.subplots(2, 6, figsize=(16, 6))
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    mat_in = x[0].squeeze().detach().cpu().numpy()
    mat_prev_ice = x[:, [2], :, :].squeeze().detach().cpu().numpy()
    mat_out = outputs[0].squeeze().detach().cpu().numpy()# [0]  # take first channel of output, this is sea ice
    mat_true = y[0].squeeze().detach().cpu().numpy()
    ax[0].imshow(mat_prev_ice)
    ax[0].set_title(f"Previous Ice")
    ax[1].imshow(mat_out)
    ax[1].set_title(f"Forecast")
    ax[2].imshow(mat_true)
    ax[2].set_title(f"Truth")
    ax[3].imshow(mat_true - mat_out, cmap="bwr")
    ax[3].set_title(f"Truth - Forecast")
    # for i in range(2):
    #     for j in range(6):
    #         if 6*i + j < 9:
    #             ax[i][j].imshow(mat_in[6*i + j])
    #             ax[i][j].set_title(f"Input Channel {6*i + j}")
    #         else:
    #             ax[i][j].imshow(mat_out)
    #             ax[i][j].set_title(f"Prediction")
    #             ax[i][j+1].imshow(mat_true)
    #             ax[i][j+1].set_title(f"Truth")
    #             ax[i][j+2].set_axis_off()
    #             break
    plt.tight_layout()

    # save the outputs
    sample_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"{sample_dir}/{timestring}_{epoch:04d}.png")
    plt.close()


def train():

    # load config hyperparameters
    config = TrainingConfig()

    # set up model and scheduler
    unet = UNet2DConditionModel(
        in_channels = 1, # config.in_channels + 1,  # condition dim + noise dim
        out_channels = config.out_channels,
        block_out_channels = (128, 256, 512, 1024),
        cross_attention_dim = config.cross_attention_dim,
        norm_num_groups=1
    )
    scheduler = PNDMScheduler(num_train_timesteps=config.num_train_timesteps)

    # unet = UNet(input_channels=config.in_channels)  # use custom unet

    # set up dataset
    train_ds = IceNetDataSetPyTorch(config.dataset_config, "train", batch_size=config.train_batch_size, shuffling=False)
    val_ds = IceNetDataSetPyTorch(config.dataset_config, "val", batch_size=config.train_batch_size, shuffling=False)
    test_ds = IceNetDataSetPyTorch(config.dataset_config, "test", batch_size=config.train_batch_size, shuffling=False)

    # configure dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=False)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config.eval_batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.eval_batch_size, shuffle=False)

    # modify prebuilt vae to be proper shape 
    # pipeline.vae.encoder.conv_in = nn.Conv2d(in_channels=config.in_channels, out_channels=config.mid_channels,
    #                                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # pipeline.vae.decoder.conv_out = nn.Conv2d(in_channels=config.mid_channels, out_channels=config.out_channels,
    #                                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # configure optimizers
    optimizer_unet = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    # lr_scheduler_unet = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer_unet,
    #     num_warmup_steps=config.lr_warmup_steps,
    #     num_training_steps=config.num_epochs
    # )
    lr_scheduler_unet = get_constant_schedule_with_warmup(
        optimizer=optimizer_unet,
        num_warmup_steps=config.lr_warmup_steps,
    )
    # lr_scheduler_unet = get_piecewise_constant_schedule(
    #     optimizer=optimizer_unet,
    #     step_rules="1:500,0.1:500,0.01"
    # )
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
    unet, optimizer_unet, lr_scheduler_unet, train_dataloader = accelerator.prepare(
        unet, optimizer_unet, lr_scheduler_unet, train_dl
    )

    # begin training loop
    global_step = 0
    for epoch in range(config.num_epochs):

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):

            with torch.autocast(device_type="cuda"):

                # unpack batch and create latents using vae
                x, y, sw = batch

                # DOWNSAMPLE
                x = F.interpolate(x, (54, 54))
                y = F.interpolate(y, (54, 54))
                sw = F.interpolate(sw, (54, 54))

                # y = torch.cat((y, torch.zeros((config.train_batch_size, 8, 432, 432), device="cuda")), dim=1)
                # latents = pipeline.vae.encode(x).latent_dist.mode()
                latents = x
                # target_latents = pipeline.vae.encode(y).latent_dist.mode()

                # add previous ice state since network is trained to predict residual
                target_latents = y - x[:, [2], :, :]

                # sw_latents = F.interpolate(sw, config.latent_size, mode="nearest")
                sw_latents = sw

                # sample noise to add to the images
                noise = torch.randn(target_latents.shape, device=target_latents.device)  # init from noise
                # noise = latents[:, [2], :, :]  # init from previous ice
                bs = target_latents.shape[0]

                # sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps, 
                    (bs,),
                    device=target_latents.device,
                    dtype=torch.int64
                )

                # add noise to clean images in forward diffusion process
                noisy_images = scheduler.add_noise(target_latents, noise, timesteps)

                # concatenate predictors onto noisy images
                # noisy_images = torch.cat((noisy_images, latents), dim=1)

                with accelerator.accumulate(unet):

                    # predict the noise residual and multiply by mask
                    noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=torch.zeros((1, 1, config.cross_attention_dim), device="cuda")).sample
                    if config.use_mask:
                        masked_noise_pred = noise_pred * sw_latents
                    else:
                        masked_noise_pred = noise_pred

                    # add previous ice state such that network is trained to predict residual
                    masked_noise_pred = noise_pred + latents[:, [2], :, :]

                    # compute mse loss on residual
                    loss = F.mse_loss(masked_noise_pred, noise)
                    accelerator.backward(loss)

                    # step optimizers
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer_unet.step()
                    lr_scheduler_unet.step()
                    optimizer_unet.zero_grad()

                # update logs and trackers
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler_unet.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        # optionally sample some demo forecassts with evaluate() and save the model
        if accelerator.is_main_process:

            if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, unet, scheduler, train_dataloader)

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

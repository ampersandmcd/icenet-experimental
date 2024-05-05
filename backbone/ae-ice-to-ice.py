# partially inspired by https://github.com/huggingface/diffusers/pull/3801/files#diff-55df4b31629a5a38c256abdc62975871d9ce56c36425b4e20304d024e107a159
import os
import torch
from diffusers import AutoencoderKL, PNDMScheduler, DiffusionPipeline
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
    in_channels = 1
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
    output_dir = "results/ae-ice-to-ice"  # the model name locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0
    use_mask = False
    dataset_config = "dataset_config.single_day.json"
    cross_attention_dim = 1
    kl_scale = 1e-8


def evaluate(config, epoch, ae, train_dataloader, accelerator):

    with torch.autocast(device_type="cuda"):

        # unpack batch
        batch = next(iter(train_dataloader))
        x, y, sw = batch

        # run through
        posterior = ae.encode(y).latent_dist
        z = posterior.mode()
        y_hat = ae.decode(z).sample

        # mask if relevant
        if config.use_mask:
            y_hat = y_hat * sw

        # compute losses
        kl_loss = posterior.kl().mean()
        mse_loss = F.mse_loss(y_hat, y, reduction="mean")
        loss = mse_loss + config.kl_scale * kl_loss

    # make plot
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    mat_in = y[0].squeeze().detach().cpu().numpy()
    mat_latent = z[0].squeeze().detach().cpu().numpy()
    mat_out = y_hat[0].squeeze().detach().cpu().numpy()
    ax[0].imshow(mat_in)
    ax[0].set_title(f"Input/True Ice")
    ax[1].imshow(mat_latent[0])
    ax[1].set_title(f"Latent Ice\n(First Channel)")
    ax[2].imshow(mat_out)
    ax[2].set_title(f"Output Ice")
    ax[3].imshow(mat_in - mat_out, cmap="bwr")
    ax[3].set_title(f"Output - Input/True")
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

    # set up model from pretrained stable diffusion vae
    ae = AutoencoderKL(
        in_channels = config.in_channels,
        out_channels = config.out_channels,
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (128, 256, 512, 1024),
        norm_num_groups=1
    )
    # ae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    # ae.encoder.conv_in = nn.Conv2d(in_channels=config.in_channels, out_channels=config.mid_channels,
    #                                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # ae.decoder.conv_out = nn.Conv2d(in_channels=config.out_channels, out_channels=config.out_channels,
    #                                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # set up dataset
    train_ds = IceNetDataSetPyTorch(config.dataset_config, "train", batch_size=config.train_batch_size, shuffling=False)
    val_ds = IceNetDataSetPyTorch(config.dataset_config, "val", batch_size=config.train_batch_size, shuffling=False)
    test_ds = IceNetDataSetPyTorch(config.dataset_config, "test", batch_size=config.train_batch_size, shuffling=False)

    # configure dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=False)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config.eval_batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.eval_batch_size, shuffle=False)

    # configure optimizers
    optimizer = torch.optim.AdamW(ae.parameters(), lr=config.learning_rate)
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
    )

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
    ae, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        ae, optimizer, lr_scheduler, train_dl
    )

    # begin training loop
    global_step = 0
    for epoch in range(config.num_epochs):

        train_loss = 0.0
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with torch.autocast(device_type="cuda"):
                with accelerator.accumulate(ae):

                    # unpack batch
                    x, y, sw = batch

                    # run through
                    posterior = ae.encode(y).latent_dist
                    z = posterior.mode()
                    y_hat = ae.decode(z).sample

                    # mask if relevant
                    if config.use_mask:
                        y_hat = y_hat * sw

                    # compute losses
                    kl_loss = posterior.kl().mean()
                    mse_loss = F.mse_loss(y_hat, y, reduction="mean")
                    loss = mse_loss + config.kl_scale * kl_loss

                    # optimizer step
                    accelerator.clip_grad_norm_(ae.parameters(), 1.0)
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # update logs and trackers
                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "mse": mse_loss.detach().item(),
                    "kl": kl_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        # optionally sample some demo forecassts with evaluate() and save the model
        if accelerator.is_main_process:

            if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, ae, train_dataloader, accelerator)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    torch.save(ae, Path(config.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pth"))


if __name__ == "__main__":
    train()

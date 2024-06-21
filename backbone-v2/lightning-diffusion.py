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
from src.utils import IceNetDataset, Visualise
from src.models import UNet, LitUNet, Generator, Discriminator, LitGAN
from src import config

# trade off speed and performance depending on gpu
torch.set_float32_matmul_precision("medium")
# torch.set_float32_matmul_precision("high")


def train_icenet(args):
    """
    Train IceNet using the arguments specified in the `args` namespace.
    :param args: Namespace of configuration parameters
    """
    # init
    pl.seed_everything(args.seed)
    
    # configure datasets and dataloaders
    dataloader_config_fpath = os.path.join(config.dataloader_config_folder, args.dataloader_config)
    train_dataset = IceNetDataset(dataloader_config_fpath, mode="train")
    val_dataset = IceNetDataset(dataloader_config_fpath, mode="val")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                                 persistent_workers=True, shuffle=False)  # IceNetDataset class handles shuffling
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                                persistent_workers=True, shuffle=False)  # IceNetDataset class handles shuffling

    # configure model
    if args.model == "unet":

        # construct unet
        model = UNet(
            input_channels=train_dataset.tot_num_channels,
            filter_size=args.filter_size,
            n_filters_factor=args.n_filters_factor,
            n_forecast_months=train_dataset.config["n_forecast_months"]
        )
        
        # configure unet loss with reduction="none" for sample weighting
        if args.criterion == "ce":
            criterion = nn.CrossEntropyLoss(reduction="none")
        elif args.criterion == "focal":
            criterion = sigmoid_focal_loss  # reduction="none" by default
        else:
            raise NotImplementedError(f"Invalid UNet loss function: {args.criterion}.")
        
        # configure PyTorch Lightning module
        lit_module = LitUNet(
            model=model,
            criterion=criterion,
            learning_rate=args.learning_rate
        )

    elif args.model == "gan":

        # construct generator
        generator = Generator(
            input_channels=train_dataset.tot_num_channels,
            filter_size=args.filter_size,
            n_filters_factor=args.n_filters_factor,
            n_forecast_months=train_dataset.config["n_forecast_months"],
            sigma=args.sigma
        )

        # construct discriminator
        discriminator = Discriminator(
            input_channels=train_dataset.tot_num_channels,
            filter_size=args.filter_size,
            n_filters_factor=args.n_filters_factor,
            n_forecast_months=train_dataset.config["n_forecast_months"],
            mode=args.discriminator_mode
        )
        
        # configure losses with reduction="none" for sample weighting
        if args.generator_fake_criterion == "ce":
            generator_fake_criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError(f"Invalid generator fake loss function: {args.criterion}.")

        if args.generator_structural_criterion == "l1":
            generator_structural_criterion = nn.L1Loss(reduction="none")
        elif args.generator_structural_criterion == "ce":
            generator_structural_criterion = nn.CrossEntropyLoss(reduction="none")
        elif args.generator_structural_criterion == "focal":
            generator_structural_criterion = sigmoid_focal_loss
        else:
            raise ValueError(f"Invalid generator structural loss function: {args.criterion}.")

        if args.discriminator_criterion == "ce":
            discriminator_criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError(f"Invalid discriminator loss function: {args.criterion}.")
        
        # configure PyTorch Lightning module
        lit_module = LitGAN(
            generator=generator,
            discriminator=discriminator,
            generator_fake_criterion=generator_fake_criterion,
            generator_structural_criterion=generator_structural_criterion,
            generator_lambda=args.generator_lambda,
            discriminator_criterion=discriminator_criterion,
            learning_rate=args.learning_rate,
            d_lr_factor=args.d_lr_factor
        )

    else:
        raise NotImplementedError(f"Invalid model architecture specified: {args.model}")

    # set up wandb logging
    wandb.init(project="icenet-gan")
    if args.name != "default":
        wandb.run.name = args.name
    wandb_logger = pl.loggers.WandbLogger(project="icenet-gan")
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
    trainer.callbacks.append(ModelCheckpoint(monitor="val_accuracy", mode="max"))
    trainer.callbacks.append(Visualise(val_dataloader, 
                                       n_to_visualise=args.n_to_visualise, 
                                       n_forecast_months=val_dataset.config["n_forecast_months"]))

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
    
    # model configurations applicable to UNet
    parser.add_argument("--criterion", default="focal", type=str, choices=["ce", "focal"],
                        help="Loss to train UNet", required=False)
    
    # model configurations applicable to GAN
    parser.add_argument("--sigma", default=1, type=float,
                        help="Noise factor to set sampling temperature in G", required=False)    
    parser.add_argument("--generator_fake_criterion", default="ce", type=str, choices=["ce"],
                        help="Loss to train GAN generator instance-level fake-out", required=False)
    parser.add_argument("--generator_structural_criterion", default="ce", type=str, choices=["ce", "focal", "l1"],
                        help="Loss to train GAN generator structural similarity", required=False)    
    parser.add_argument("--generator_lambda", default=500, type=float,
                        help="Trade off between instance-level fake-out and structural loss", required=False)    
    parser.add_argument("--discriminator_mode", default="forecast", type=str, choices=["forecast", "onestep"],
                        help="Whether D should classify entire forecast as real/fake or"
                             "each timestep separately as real/fake", required=False)
    parser.add_argument("--discriminator_criterion", default="ce", type=str, choices=["ce"],
                        help="Loss to train GAN discriminator", required=False)
    parser.add_argument("--d_lr_factor", default=1, type=float,
                        help="Factor by which to multiply G learning rate for use on D", required=False)    
    
    # model configurations applicable to both UNet and GAN
    parser.add_argument("--dataloader_config", default="2023_06_24_1235_icenet_gan.json", type=str,
                        help="Filename of dataloader_config.json file")
    parser.add_argument("--filter_size", default=3, type=int, help="Filter (kernel) size for convolutional layers")
    parser.add_argument("--n_filters_factor", default=1.0, type=float,
                        help="Scale factor with which to modify number of convolutional filters")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for UNet/Generator")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    # hardware configurations applicable to both UNet and GAN
    parser.add_argument("--accelerator", default="auto", type=str, help="PytorchLightning training accelerator")
    parser.add_argument("--devices", default=1, type=int, help="PytorchLightning number of devices to run on")
    parser.add_argument("--n_workers", default=8, type=int, help="Number of workers in dataloader")
    parser.add_argument("--precision", default=16, type=int, choices=[32, 16], help="Precision for training")
    
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

"""
Adapted from Tom Andersson's https://github.com/tom-andersson/icenet-paper/blob/main/icenet/models.py
Modified to be modular and importable.
Modified from Tensorflow to PyTorch.
Modified to support PyTorch Lightning.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from src import config
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from torchmetrics import JaccardIndex, Dice, Accuracy, Precision, Recall, F1Score  # classification
from src.metrics import IceNetAccuracy, SIEError


### Custom layers:
# --------------------------------------------------------------------


class TemperatureScale(nn.Module):
    '''
    Implements the temperature scaling layer for probability calibration,
    as introduced in Guo 2017 (http://proceedings.mlr.press/v70/guo17a.html).
    '''
    def __init__(self, **kwargs):
        super(TemperatureScale, self).__init__()
        self.temp = nn.Parameter(data=torch.Tensor([1.0]), requires_grad=False)

    def call(self, inputs):
        ''' Divide the input logits by the T value. '''
        return torch.divide(inputs, self.temp)

    def get_config(self):
        ''' For saving and loading networks with this custom layer. '''
        return {'temp': self.temp.numpy()}


### Network architectures:
# --------------------------------------------------------------------


class UNet(nn.Module):
    """
    An implementation of a UNet for pixelwise classification.
    """
    
    def __init__(self,
                 input_channels, 
                 filter_size=3, 
                 n_filters_factor=1, 
                 n_forecast_months=6, 
                 use_temp_scaling=False,
                 n_output_classes=3,
                **kwargs):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.filter_size = filter_size
        self.n_filters_factor = n_filters_factor
        self.n_forecast_months = n_forecast_months
        self.use_temp_scaling = use_temp_scaling
        self.n_output_classes = n_output_classes

        self.conv1a = nn.Conv2d(in_channels=input_channels, 
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv1b = nn.Conv2d(in_channels=int(128*n_filters_factor),
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn1 = nn.BatchNorm2d(num_features=int(128*n_filters_factor))

        self.conv2a = nn.Conv2d(in_channels=int(128*n_filters_factor),
                                out_channels=int(256*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv2b = nn.Conv2d(in_channels=int(256*n_filters_factor),
                                out_channels=int(256*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn2 = nn.BatchNorm2d(num_features=int(256*n_filters_factor))

        self.conv3a = nn.Conv2d(in_channels=int(256*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv3b = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn3 = nn.BatchNorm2d(num_features=int(512*n_filters_factor))

        self.conv4a = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv4b = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn4 = nn.BatchNorm2d(num_features=int(512*n_filters_factor))

        self.conv5a = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(1024*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv5b = nn.Conv2d(in_channels=int(1024*n_filters_factor),
                                out_channels=int(1024*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn5 = nn.BatchNorm2d(num_features=int(1024*n_filters_factor))

        self.conv6a = nn.Conv2d(in_channels=int(1024*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv6b = nn.Conv2d(in_channels=int(1024*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv6c = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn6 = nn.BatchNorm2d(num_features=int(512*n_filters_factor))

        self.conv7a = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv7b = nn.Conv2d(in_channels=int(1024*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv7c = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(512*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn7 = nn.BatchNorm2d(num_features=int(512*n_filters_factor))

        self.conv8a = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(256*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv8b = nn.Conv2d(in_channels=int(512*n_filters_factor),
                                out_channels=int(256*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv8c = nn.Conv2d(in_channels=int(256*n_filters_factor),
                                out_channels=int(256*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn8 = nn.BatchNorm2d(num_features=int(256*n_filters_factor))

        self.conv9a = nn.Conv2d(in_channels=int(256*n_filters_factor),
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv9b = nn.Conv2d(in_channels=int(256*n_filters_factor),
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv9c = nn.Conv2d(in_channels=int(128*n_filters_factor),
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")  # no batch norm on last layer

        self.final_conv = nn.Conv2d(in_channels=int(128*n_filters_factor),
                                    out_channels=n_output_classes*n_forecast_months,
                                    kernel_size=filter_size,
                                    padding="same")
        
    def forward(self, x):

        # transpose from shape (b, h, w, c) to (b, c, h, w) for pytorch conv2d layers
        x = torch.movedim(x, -1, 1)  # move c from last to second dim

        # run through network
        conv1 = self.conv1a(x)  # input to 128
        conv1 = F.relu(conv1)
        conv1 = self.conv1b(conv1)  # 128 to 128
        conv1 = F.relu(conv1)
        bn1 = self.bn1(conv1)
        pool1 = F.max_pool2d(bn1, kernel_size=(2, 2))

        conv2 = self.conv2a(pool1)  # 128 to 256
        conv2 = F.relu(conv2)
        conv2 = self.conv2b(conv2)  # 256 to 256
        conv2 = F.relu(conv2)
        bn2 = self.bn2(conv2)
        pool2 = F.max_pool2d(bn2, kernel_size=(2, 2))

        conv3 = self.conv3a(pool2)  # 256 to 512
        conv3 = F.relu(conv3)
        conv3 = self.conv3b(conv3)  # 512 to 512
        conv3 = F.relu(conv3)
        bn3 = self.bn3(conv3)
        pool3 = F.max_pool2d(bn3, kernel_size=(2, 2))

        conv4 = self.conv4a(pool3)  # 512 to 512
        conv4 = F.relu(conv4)
        conv4 = self.conv4b(conv4)  # 512 to 512
        conv4 = F.relu(conv4)
        bn4 = self.bn4(conv4)
        pool4 = F.max_pool2d(bn4, kernel_size=(2, 2))

        conv5 = self.conv5a(pool4)  # 512 to 1024
        conv5 = F.relu(conv5)
        conv5 = self.conv5b(conv5)  # 1024 to 1024
        conv5 = F.relu(conv5)
        bn5 = self.bn5(conv5)

        up6 = F.upsample(bn5, scale_factor=2, mode="nearest")
        up6 = self.conv6a(up6)  # 1024 to 512
        up6 = F.relu(up6)
        merge6 = torch.cat([bn4, up6], dim=1) # 512 and 512 to 1024 along c dimension
        conv6 = self.conv6b(merge6)  # 1024 to 512
        conv6 = F.relu(conv6)
        conv6 = self.conv6c(conv6)  # 512 to 512
        conv6 = F.relu(conv6)
        bn6 = self.bn6(conv6)

        up7 = F.upsample(bn6, scale_factor=2, mode="nearest")
        up7 = self.conv7a(up7)  # 1024 to 512
        up7 = F.relu(up7)
        merge7 = torch.cat([bn3, up7], dim=1) # 512 and 512 to 1024 along c dimension
        conv7 = self.conv7b(merge7)  # 1024 to 512
        conv7 = F.relu(conv7)
        conv7 = self.conv7c(conv7)  # 512 to 512
        conv7 = F.relu(conv7)
        bn7 = self.bn7(conv7)

        up8 = F.upsample(bn7, scale_factor=2, mode="nearest")
        up8 = self.conv8a(up8)  # 512 to 256
        up8 = F.relu(up8)
        merge8 = torch.cat([bn2, up8], dim=1) # 256 and 256 to 512 along c dimension
        conv8 = self.conv8b(merge8)  # 512 to 256
        conv8 = F.relu(conv8)
        conv8 = self.conv8c(conv8)  # 256 to 256
        conv8 = F.relu(conv8)
        bn8 = self.bn8(conv8)

        up9 = F.upsample(bn8, scale_factor=2, mode="nearest")
        up9 = self.conv9a(up9)  # 256 to 128
        up9 = F.relu(up9)
        merge9 = torch.cat([bn1, up9], dim=1) # 128 and 128 to 256 along c dimension
        conv9 = self.conv9b(merge9)  # 256 to 128
        conv9 = F.relu(conv9)
        conv9 = self.conv9c(conv9)  # 128 to 128
        conv9 = F.relu(conv9)  # no batch norm on last layer
 
        final_layer_logits = self.final_conv(conv9)

        # transpose from shape (b, c, h, w) back to (b, h, w, c) to align with training data
        final_layer_logits = torch.movedim(final_layer_logits, 1, -1)  # move c from second to final dim
        b, h, w, c = final_layer_logits.shape

        # unpack c=classes*months dimension into classes, months as separate dimensions
        final_layer_logits = final_layer_logits.reshape((b, h, w, self.n_output_classes, self.n_forecast_months))

        if self.use_temp_scaling:
            final_layer_logits_scaled = TemperatureScale()(final_layer_logits)
            output = F.softmax(final_layer_logits_scaled, dim=-2)  # apply over n_output_classes dimension
        else:
            output = F.softmax(final_layer_logits, dim=-2)  # apply over n_output_classes dimension

        return output  # shape (b, h, w, c, t)

### PyTorch Lightning modules:
# --------------------------------------------------------------------

class LitDiffusion(pl.LightningModule):
    """
    A LightningModule wrapping the diffusion implementation of IceNet.
    """
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 generator_fake_criterion: callable,
                 generator_structural_criterion: callable,
                 generator_lambda: float,
                 discriminator_criterion: callable,
                 learning_rate: float,
                 d_lr_factor: float):
        """
        Construct a UNet LightningModule.
        Note that we keep hyperparameters separate from dataloaders to prevent data leakage at test time.
        :param generator: PyTorch model to generate forecasts
        :param discriminator: PyTorch model to discriminate between real forecasts (observations) and fake forecasts
        :param generator_fake_criterion: Instance-level loss function for G instantiated with reduction="none"
        :param generator_structural_criterion: Structural-level loss function for G instantiated with reduction="none"
        :param generator_lambda: Parameter to trade off between instance and structure loss for G
        :param discriminator_criterion: Instance-level loss function for D instantiated with reduction="none"
        :param learning_rate: Float learning rate for our optimiser
        :param learning_rate: Float factor to adjust D learning rate so it is balanced with G
        """
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_fake_criterion = generator_fake_criterion
        self.generator_structural_criterion = generator_structural_criterion
        self.generator_lambda = generator_lambda
        self.discriminator_criterion = discriminator_criterion
        self.learning_rate = learning_rate
        self.d_lr_factor = d_lr_factor
        self.n_output_classes = generator.n_output_classes  # this should be a property of the network

        # manually control optimisation
        self.automatic_optimization = False

        # evaluation metrics
        # for details see: https://torchmetrics.readthedocs.io/en/stable/
        # note https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
        # > These results mean that in multi-class classification cases where each observation has a single label, 
        # > the micro-F1, micro-precision, micro-recall, and accuracy share the same value (i.e., 0.60 in this example).
        # > And this explains why the classification report only needs to display a single accuracy value since 
        # > micro-F1, micro-precision, and micro-recall also have the same value.
        # self.metrics = MetricCollection({
        #     "val_jaccard": JaccardIndex(task="multiclass", num_classes=self.n_output_classes),
        #     "val_dice": Dice(task="multiclass", num_classes=self.n_output_classes),
        #     "val_accuracy": Accuracy(task="multiclass", num_classes=self.n_output_classes, average="micro"),
        #     "val_macro_accuracy": Accuracy(task="multiclass", num_classes=self.n_output_classes, average="macro"),
        #     "val_weighted_accuracy": Accuracy(task="multiclass", num_classes=self.n_output_classes, average="weighted"),
        #     "val_macro_precision": Precision(task="multiclass", num_classes=self.n_output_classes, average="macro"),
        #     "val_weighted_precision": Precision(task="multiclass", num_classes=self.n_output_classes, average="weighted"),
        #     "val_macro_recall": Recall(task="multiclass", num_classes=self.n_output_classes, average="macro"),
        #     "val_weighted_recall": Recall(task="multiclass", num_classes=self.n_output_classes, average="weighted"),
        #     "val_macro_f1": F1Score(task="multiclass", num_classes=self.n_output_classes, average="macro"),
        #     "val_weighted_f1": F1Score(task="multiclass", num_classes=self.n_output_classes, average="weighted"),
        #     # TODO: add more metrics and compare to original icenet
        # })
        metrics = {
            "val_accuracy": IceNetAccuracy(leadtimes_to_evaluate=[0, 1, 2, 3, 4, 5]),
            "val_sieerror": SIEError(leadtimes_to_evaluate=[0, 1, 2, 3, 4, 5])
        }
        for i in range(6):
            metrics[f"val_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
            metrics[f"val_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.metrics = MetricCollection(metrics)

        test_metrics = {
            "test_accuracy": IceNetAccuracy(leadtimes_to_evaluate=[0, 1, 2, 3, 4, 5]),
            "test_sieerror": SIEError(leadtimes_to_evaluate=[0, 1, 2, 3, 4, 5])
        }
        for i in range(6):
            test_metrics[f"test_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
            test_metrics[f"test_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.test_metrics = MetricCollection(test_metrics)

        self.save_hyperparameters(ignore=["generator", "discriminator", 
                                          "generator_fake_criterion",
                                          "generator_structural_criterion",
                                          "discriminator_criterion"])

    def forward(self, x):
        """
        Implement forward function.
        :param x: Inputs to model.
        :return: Outputs of model.
        """
        return self.generator(x)

    def training_step(self, batch, batch_idx: int):
        """
        Perform a pass through a batch of training data.
        Use Pix2Pix loss function with cross-entropy and L1 structure losses.
        See: https://www.tensorflow.org/tutorials/generative/pix2pix#define_the_generator_loss
        :param batch: Batch of input, output, weight triplets
        :param batch_idx: Index of batch
        :return: Loss from this batch of data for use in backprop
        """
        x, y, sample_weight = batch
        g_opt, d_opt = self.optimizers()

        ####################
        # Train Generator
        ####################
        self.toggle_optimizer(g_opt)

        # generate forecasts
        fake_forecasts = self.generator(x)

        # pass fake forecasts to discriminator
        d_fake_forecasts = self.discriminator(fake_forecasts, sample_weight)

        # try to fake out discriminator where real==1 and fake==0
        g_fake_loss = self.generator_fake_criterion(d_fake_forecasts, torch.ones_like(d_fake_forecasts))
        g_fake_loss = torch.mean(g_fake_loss)  # weight real/fake loss equally on each instance

        # compute loss to preserve structural similarity of forecasts
        if isinstance(self.generator_structural_criterion, nn.CrossEntropyLoss):  # requires int class encoding
            g_structural_loss = self.generator_structural_criterion(fake_forecasts.movedim(-2, 1), y.argmax(-2).long())
        else:  # requires one-hot encoding
            g_structural_loss = self.generator_structural_criterion(fake_forecasts.movedim(-2, 1), y.movedim(-2, 1))
        g_structural_loss = torch.mean(g_structural_loss * sample_weight.movedim(-2, 1))  # weight spatially

        # sum losses with hyperparameter lambda for generator's total loss
        g_loss = g_fake_loss + self.generator_lambda * g_structural_loss
        self.log("g_train_loss", g_loss, prog_bar=True, sync_dist=True)
        self.log("g_train_loss_fake", g_fake_loss, sync_dist=True)
        self.log("g_train_loss_structural", g_structural_loss, sync_dist=True)

        # manually step generator optimiser
        self.manual_backward(g_loss)
        g_opt.step()
        g_opt.zero_grad()
        self.untoggle_optimizer(g_opt)
        
        #####################
        # Train Discriminator
        #####################
        self.toggle_optimizer(d_opt)

        # try to detect real forecasts (observations) where real==1 and fake==0
        d_real_forecasts = self.discriminator(y, sample_weight)
        d_real_loss = self.discriminator_criterion(d_real_forecasts, torch.ones_like(d_real_forecasts))
        d_real_loss = d_real_loss.mean()  # weight real/fake loss equally on each instance

        # generate fake forecasts
        fake_forecasts = self.generator(x)

        # pass fake forecasts to discriminator
        d_fake_forecasts = self.discriminator(fake_forecasts, sample_weight)

        # try to detect fake forecasts where real==1 and fake==0
        d_fake_loss = self.discriminator_criterion(d_fake_forecasts, torch.zeros_like(d_fake_forecasts))
        d_fake_loss = torch.mean(d_fake_loss)  # weight real/fake loss equally on each instance

        # sum losses with equal weight
        d_loss = d_real_loss + d_fake_loss
        self.log("d_train_loss", d_loss, prog_bar=True, sync_dist=True)
        self.log("d_train_loss_real", d_real_loss, sync_dist=True)
        self.log("d_train_loss_fake", d_fake_loss, sync_dist=True)

        # manually step discriminator optimiser
        self.manual_backward(d_loss)
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)        

    def validation_step(self, batch, batch_idx):
        x, y, sample_weight = batch

        ####################
        # Validate Generator
        ####################
            
        # generate forecasts
        fake_forecasts = self.generator(x)

        # pass fake forecasts to discriminator
        d_fake_forecasts = self.discriminator(fake_forecasts, sample_weight)

        # try to fake out discriminator where real==1 and fake==0
        g_fake_loss = self.generator_fake_criterion(d_fake_forecasts, torch.ones_like(d_fake_forecasts))
        g_fake_loss = torch.mean(g_fake_loss)  # weight real/fake loss equally on each instance

        # compute loss to preserve structural similarity of forecasts
        if isinstance(self.generator_structural_criterion, nn.CrossEntropyLoss):  # requires int class encoding
            g_structural_loss = self.generator_structural_criterion(fake_forecasts.movedim(-2, 1), y.argmax(-2).long())
        else:  # requires one-hot encoding
            g_structural_loss = self.generator_structural_criterion(fake_forecasts.movedim(-2, 1), y.movedim(-2, 1))
        g_structural_loss = torch.mean(g_structural_loss * sample_weight.movedim(-2, 1))  # weight spatially

        # sum losses with hyperparameter lambda for generator's total loss
        g_loss = g_fake_loss + self.generator_lambda * g_structural_loss

        # log at epoch-level
        self.log("g_val_loss", g_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("g_val_loss_fake", g_fake_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("g_val_loss_structural", g_structural_loss, on_step=False, on_epoch=True, sync_dist=True)

        ########################
        # Validate Discriminator
        ########################
    
        # try to detect real forecasts (observations) where real==1 and fake==0
        d_real_forecasts = self.discriminator(y, sample_weight)
        d_real_loss = self.discriminator_criterion(d_real_forecasts, torch.ones_like(d_real_forecasts))
        d_real_loss = d_real_loss.mean()  # weight real/fake loss equally on each instance

        # pass fake forecasts to discriminator
        d_fake_forecasts = self.discriminator(fake_forecasts, sample_weight)

        # try to detect fake forecasts where real==1 and fake==0
        d_fake_loss = self.discriminator_criterion(d_fake_forecasts, torch.zeros_like(d_fake_forecasts))
        d_fake_loss = torch.mean(d_fake_loss)  # weight real/fake loss equally on each instance

        # sum losses with equal weight
        d_loss = d_real_loss + d_fake_loss

        # log at epoch-level
        self.log("d_val_loss", d_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("d_val_loss_real", d_real_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("d_val_loss_fake", d_fake_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        ########################
        # Forecast Metrics
        ########################

        # y and y_hat are shape (b, h, w, c, t) but loss expects (b, c, h, w, t)
        # note that criterion needs reduction="none" for weighting to work
        y_hat_pred = fake_forecasts.argmax(dim=-2).long()  # argmax over c where shape is (b, h, w, c, t)
        self.metrics.update(y_hat_pred, y.argmax(dim=-2).long(), sample_weight.squeeze(dim=-2))  # all shape (b, h, w, t)

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y, sample_weight = batch

        ####################
        # Test Generator
        ####################
            
        # generate forecasts
        fake_forecasts = self.generator(x)

        # pass fake forecasts to discriminator
        d_fake_forecasts = self.discriminator(fake_forecasts, sample_weight)

        # try to fake out discriminator where real==1 and fake==0
        g_fake_loss = self.generator_fake_criterion(d_fake_forecasts, torch.ones_like(d_fake_forecasts))
        g_fake_loss = torch.mean(g_fake_loss)  # weight real/fake loss equally on each instance

        # compute loss to preserve structural similarity of binary forecasts
        g_structural_loss = self.generator_structural_criterion(fake_forecasts, y)
        if isinstance(self.generator_structural_criterion, nn.CrossEntropyLoss):  # requires int class encoding
            g_structural_loss = self.generator_structural_criterion(fake_forecasts.movedim(-2, 1), y.argmax(-2).long())
        else:  # requires one-hot encoding
            g_structural_loss = self.generator_structural_criterion(fake_forecasts.movedim(-2, 1), y.movedim(-2, 1))
        g_structural_loss = torch.mean(g_structural_loss * sample_weight.movedim(-2, 1))  # weight spatially

        # sum losses with hyperparameter lambda for generator's total loss
        g_loss = g_fake_loss + self.generator_lambda * g_structural_loss

        # log and continue
        self.log("g_test_loss", g_loss, on_step=False, on_epoch=True, sync_dist=True)  # epoch-level loss

        ########################
        # Test Discriminator
        ########################
    
        # try to detect real forecasts (observations) where real==1 and fake==0
        d_real_forecasts = self.discriminator(y, sample_weight)
        d_real_loss = self.discriminator_criterion(d_real_forecasts, torch.ones_like(d_real_forecasts),)
        d_real_loss = d_real_loss.mean()  # weight real/fake loss equally on each instance

        # pass fake forecasts to discriminator
        d_fake_forecasts = self.discriminator(fake_forecasts, sample_weight)

        # try to detect fake forecasts where real==1 and fake==0
        d_fake_loss = self.discriminator_criterion(d_fake_forecasts, torch.zeros_like(d_fake_forecasts))
        d_fake_loss = torch.mean(d_fake_loss)  # weight real/fake loss equally on each instance

        # sum losses with equal weight
        d_loss = d_real_loss + d_fake_loss

        # log and return to optimiser
        self.log("d_test_loss", d_loss, on_step=False, on_epoch=True, sync_dist=True)  # epoch-level loss
        
        ########################
        # Forecast Metrics
        ########################

        # y and y_hat are shape (b, h, w, c, t) but loss expects (b, c, h, w, t)
        # note that criterion needs reduction="none" for weighting to work
        y_hat_pred = fake_forecasts.argmax(dim=-2).long()  # argmax over c where shape is (b, h, w, c, t)
        self.metrics.update(y_hat_pred, y.argmax(dim=-2).long(), sample_weight.squeeze(dim=-2))  # shape (b, h, w, t)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.test_metrics.reset()

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate * self.d_lr_factor)
        return [g_opt, d_opt], []  # add schedulers to second list if desired


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

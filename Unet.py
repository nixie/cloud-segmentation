import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torchmetrics
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl

from dataset import CloudDataset


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def down(in_channels, out_channels):
    return nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_channels, out_channels)
    )

class up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranpose2d(in_channels // 2, in_channels // 2,
                                        kernel_size=2, stride=2)

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [?, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1) ## why 1?
        return self.conv(x)

class Unet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, inference_threshold: float = 0.5):
        super(Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True
        self.inference_threshold = inference_threshold

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def _forward(self, x):  # x [NCHW]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

    def forward(self, x):
      return self._forward(x)

    def infer(self, x):
      y = self._forward(x)
      probs = torch.sigmoid(y)
      decisions = (probs > self.inference_threshold).int()
      return decisions.squeeze(1)

    def training_step(self, batch, batch_nb):
        x, y, ix = batch
        cropping = transforms.RandomCrop.get_params(x, [224,224])
        x = TF.crop(x, *cropping)
        y = TF.crop(y, *cropping)

        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)

        sch = self.lr_schedulers()
        sch.step()

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y, ix = batch
        cropping = transforms.RandomCrop.get_params(x, [224,224])
        x = TF.crop(x, *cropping)
        y = TF.crop(y, *cropping)

        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)

        y_hat_probs = torch.sigmoid(y_hat)
        auroc = torchmetrics.functional.classification.binary_auroc(y_hat_probs, y)

        prec = torchmetrics.functional.classification.binary_precision(y_hat_probs, y)
        rec = torchmetrics.functional.classification.binary_recall(y_hat_probs, y)

        self.log_dict({'loss/val': loss,
                       'metrics/val_auroc':auroc,
                       'metrics/val_prec': prec,
                       'metrics/val_rec': rec,
                       }, sync_dist=True)

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
              'scheduler': ExponentialLR(optimizer, 0.999),
              'interval': 'step'  # called after each training step
            },
        }



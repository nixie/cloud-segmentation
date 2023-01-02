import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
import  pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import CloudDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def main(hparams):
    model = Unet(hparams.n_channels, hparams.n_classes)

    dataset = CloudDataset()
    n_val = int(len(dataset) * 0.05)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))



    torch.manual_seed(123)  # for random transforms

    train_loader = DataLoader(train_ds, batch_size=16, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True, shuffle=False)

    os.makedirs(hparams.log_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hparams.log_dir, 'checkpoints'),
        save_last=True,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=True,
    )

    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(save_dir=os.path.join(hparams.log_dir, 'logs'), flush_secs=10)

    # training
    trainer = pl.Trainer(devices=1,
                         accelerator='gpu',
                         logger=logger,
                         max_epochs=1000,
                         precision=16,
                         log_every_n_steps=1,
                         callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--n_channels', type=int, default=4)
    parser.add_argument('--n_classes', type=int, default=1)
    hparams = parser.parse_args()

    main(hparams)

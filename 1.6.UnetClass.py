from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from utils.dataset import *
from utils.callbacks import DiceCallback as MyDiceCallbak, IouCallback as MyIouCallback, AccuracyCallback

from catalyst.dl import SupervisedRunner, DiceCallback, IouCallback, AUCCallback
from catalyst.utils import set_global_seed, prepare_cudnn

import segmentation_models_pytorch as smp

from utils.mobilenetv3 import mobilenetv3

from tqdm import tqdm
import os

prepare_cudnn(True, True)
set_global_seed(0)

NAME = '1.6.mobilenet_multi'
LOGDIR = f"./logdir/{NAME}"

# Train binary
# ------------------------------------------------------------------------------
logdir = os.path.join(LOGDIR, 'binary/')
num_epochs = 40

FP16 = True

batch_size = 24
default_batch_size = 8

lr = 1e-4 * batch_size / default_batch_size
weight_decay = 1e-5
momentum = 0.9

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 6

model = mobilenetv3(4)

# Dataloaders
train, val = get_train_val_dataloaders(df='dataset/train.csv',
                                       data_folder='dataset/train_images',
                                       mean=mean,
                                       std=std,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=False,
                                       binary=False,
                                       multi=True,
                                       hard_transforms=True)
loaders = {"train": train, "valid": val}

# Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

# Train
runner = SupervisedRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True,
    callbacks=[
        AUCCallback(),
        AccuracyCallback(threshold=0.5),
    ],
    fp16=FP16,
)
del model, train, val, criterion, optimizer, scheduler, runner
# ------------------------------------------------------------------------------


# # Train segmentation
# # ------------------------------------------------------------------------------
# num_epochs = 50
# encoder = 'resnet50'
# logdir = os.path.join(LOGDIR, 'seg/')
#
# FP16 = True
#
# batch_size = 8
# default_batch_size = 8
#
# lr = 1e-4 * batch_size / default_batch_size
# weight_decay = 1e-5
# momentum = 0.9
#
# # Dataloaders
# train, val = get_train_val_dataloaders(df='dataset/train.csv',
#                                        data_folder='dataset/train_images',
#                                        mean=mean,
#                                        std=std,
#                                        batch_size=batch_size,
#                                        num_workers=num_workers,
#                                        pin_memory=False,
#                                        binary=False,
#                                        hard_transforms=True,
#                                        only_has_mask=True)
# loaders = {"train": train, "valid": val}
#
# # Model
# model = smp.Unet(encoder, encoder_weights='imagenet', classes=4, activation=None)
#
# # Optimizer
# # criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCEWithLogitsLoss()
# optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)
#
# # Train
# runner = SupervisedRunner()
# runner.train(
#     model=model,
#     criterion=criterion,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     loaders=loaders,
#     logdir=logdir,
#     num_epochs=num_epochs,
#     verbose=True,
#     callbacks=[
#         DiceCallback(threshold=0.5, prefix='catalyst_dice'),
#         IouCallback(threshold=0.5, prefix='catalyst_iou'),
#         MyDiceCallbak(threshold=0.5),
#         MyIouCallback(threshold=0.5),
#     ],
#     fp16=FP16,
# )
#
# del train, optimizer
#
# optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#
# runner.train(
#     model=model,
#     criterion=criterion,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     loaders={"train": val, "valid": val},
#     logdir=logdir,
#     num_epochs=num_epochs,
#     verbose=True,
#     callbacks=[
#         DiceCallback(threshold=0.5, prefix='catalyst_dice'),
#         IouCallback(threshold=0.5, prefix='catalyst_iou'),
#         MyDiceCallbak(threshold=0.5),
#         MyIouCallback(threshold=0.5),
#     ],
#     fp16=FP16,
# )
# # ------------------------------------------------------------------------------
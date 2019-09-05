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

NAME = '1.2.resnet50_binary_diceloss'
LOGDIR = f"./logdir/{NAME}"

# Train binary
# ------------------------------------------------------------------------------
logdir = os.path.join(LOGDIR, 'binary/')
num_epochs = 20

FP16 = True

batch_size = 24
default_batch_size = 8

lr = 1e-4 * batch_size / default_batch_size
weight_decay = 1e-5
momentum = 0.9

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 6


model = mobilenetv3(1)

# Dataloaders
train, val = get_train_val_dataloaders(df='dataset/train.csv',
                                       data_folder='dataset/train_images',
                                       mean=mean,
                                       std=std,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=False,
                                       binary=True)
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
# ------------------------------------------------------------------------------


# Prepare data for segmentation
# ==============================================================================
df = read_dataset('dataset/train.csv', 'dataset/train_images')
dataloader = get_dataloader(df, get_inference_transforms(mean, std),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            phase='valid',
                            catalyst=False,
                            pin_memory=False,
                            binary=True)

results = runner.predict_loader(dataloader, resume=os.path.join(logdir, "checkpoints/best.pth"))
results = torch.from_numpy(results)

labels = []
for images, label in tqdm(dataloader, total=len(dataloader), desc='Inference binary'):
    labels.append(label)
labels = torch.cat(labels)

thresholds = {}
for i in np.arange(0.05, 1, 0.05):
    accuracy = AccuracyCallback.accuracy(results, labels, threshold=i)
    thresholds[i] = accuracy

thresholds = list(thresholds.items())
best_threshold = sorted(thresholds, reverse=True, key=lambda x: x[1])[0][0]

df = df.loc[(results > best_threshold).numpy().flatten()]

print(best_threshold)
with open(os.path.join(logdir, 'best_thres.txt'), 'w') as file:
    file.write(str(best_threshold))
# ==============================================================================

# Train segmentation
# ------------------------------------------------------------------------------
num_epochs = 50
encoder = 'resnet50'
logdir = os.path.join(LOGDIR, 'seg/')

FP16 = True

batch_size = 8
default_batch_size = 8

lr = 1e-4 * batch_size / default_batch_size
weight_decay = 1e-5
momentum = 0.9

# Dataloaders
train, val = get_train_val_dataloaders(df=df,
                                       mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225),
                                       batch_size=batch_size,
                                       num_workers=6,
                                       pin_memory=False)
loaders = {"train": train, "valid": val}

# Model
model = smp.Unet(encoder, encoder_weights='imagenet', classes=4, activation=None)

# Optimizer
# criterion = nn.BCEWithLogitsLoss()
criterion = smp.utils.losses.DiceLoss()
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
        DiceCallback(threshold=0.5, prefix='catalyst_dice'),
        IouCallback(threshold=0.5, prefix='catalyst_iou'),
        MyDiceCallbak(threshold=0.5),
        MyIouCallback(threshold=0.5),
    ],
    fp16=FP16,
)
# ------------------------------------------------------------------------------
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dataset import get_train_val_dataloaders
from utils.callbacks import DiceCallback as MyDiceCallbak, IouCallback as MyIouCallback

from catalyst.dl import SupervisedRunner, DiceCallback, IouCallback
from catalyst.utils import set_global_seed, prepare_cudnn

import segmentation_models_pytorch as smp

prepare_cudnn(True, True)
set_global_seed(0)

NAME = '1.1.resnet50_full_200'
logdir = f"./logdir/{NAME}"
num_epochs = 200
encoder = 'resnet50'

FP16 = True

batch_size = 8
default_batch_size = 8

lr = 1e-4 * batch_size / default_batch_size
weight_decay = 1e-5
momentum = 0.9

# Dataloaders
train, val = get_train_val_dataloaders(df='dataset/train.csv',
                                       data_folder='dataset/train_images',
                                       mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225),
                                       batch_size=batch_size,
                                       num_workers=6,
                                       pin_memory=False,
                                       full_train=True)
loaders = {"train": train, "valid": val}

# Model
model = smp.Unet(encoder, encoder_weights='imagenet', classes=4, activation=None)

# Optimizer
criterion = nn.BCEWithLogitsLoss()
# criterion = smp.utils.losses.DiceLoss()
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

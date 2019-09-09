from utils.dataset import get_dataloader, get_inference_transforms, read_dataset
from utils.callbacks import DiceCallback

import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp


def find_best_threshold(range, model, dataloader):
    dice = {i: [] for i in range}
    metric = DiceCallback(threshold=0)

    for images, masks in tqdm(dataloader, total=len(dataloader)):
        masks = masks.cuda()
        images = images.cuda()
        result = model(images)
        result = result.detach()
        for threshold in list(dice.keys()):
            dice[threshold].append(
                metric.dice(result, masks, threshold=threshold))

    dice = [(key, np.mean(item)) for key, item in dice.items()]
    dice = sorted(dice, reverse=True, key=lambda x: x[1])

    result = dice[0][0]
    print(f'Best threshold: {dice[0][0]}: {dice[0][1]:.5f}')

    return result


if __name__ == '__main__':
    # Get dataset
    transforms = get_inference_transforms(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225),)

    df = read_dataset('../dataset/train.csv',
                      '../dataset/train_images',)
    df = df.sample(500)
    dataloader = get_dataloader(df, transforms,
                                batch_size=6,
                                shuffle=False,
                                num_workers=6,
                                phase='valid',
                                catalyst=False,
                                pin_memory=False,
                                binary=False)

    # Load model
    model = smp.Unet('resnet50', encoder_weights='imagenet', classes=4, activation=None).cuda().eval()
    state = torch.load('/home/druzhinin/HDD/kaggle/kaggle_severstal/logdir/1.1.resnet50_full_200/checkpoints/best.pth')
    model.load_state_dict(state['model_state_dict'])
    del state
    model = model.eval()


    # Find best threshold
    best_thres = find_best_threshold(np.arange(0.05, 1, 0.05), model, dataloader)

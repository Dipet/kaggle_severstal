from utils.dataset import get_dataloader, get_inference_transforms, read_dataset
from utils.callbacks import DiceCallback

import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from ternausnet.models import UNet16
import cv2 as cv


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


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv.threshold(probability, threshold, 1, cv.THRESH_BINARY)[1]
    num_component, component = cv.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions

from joblib import Parallel, delayed

def _post_proc(result, thres, min_size):
    r = []
    for i in result:
        r.append(post_process(i, thres, min_size))
    return np.stack(r, axis=0)


def find_best_threshold_area_and_proba(proba_range, area_range, model, dataloader):
    dice = {}
    for p in proba_range:
        for a in area_range:
            dice[(p, a)] = []

    metric = DiceCallback(threshold=0)

    sigmoid = torch.nn.Sigmoid()

    for images, masks in tqdm(dataloader, total=len(dataloader)):
        masks = masks.cuda()
        images = images.cuda()
        result = model(images)
        result = sigmoid(result)
        result = result.detach().cpu().numpy()
        masks = masks.detach().cpu()
        for key in list(dice.keys()):
            proba, area = key

            _result = []
            for r in result:
                _r = []
                for i in r:
                    _r.append(post_process(i, proba, area))
                _r = np.stack(_r, axis=0)
                _result.append(_r)
            _result = np.stack(_result, axis=0)
            _result = torch.from_numpy(_result)

            dice[key].append(metric.dice(_result, masks, threshold=proba, activation=None))

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
    df = df.sample(10)
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
    state = torch.load('/home/druzhinin/HDD/kaggle/kaggle_severstal/logdir/1.1.resnet50_hard_transforms/checkpoints/best.pth')
    # model = UNet16(4, pretrained=True).cuda().eval()
    # state = torch.load('/home/druzhinin/HDD/kaggle/kaggle_severstal/logdir/1.5.ternausnet/checkpoints/best.pth')
    model.load_state_dict(state['model_state_dict'])
    del state
    model = model.eval()


    # Find best threshold
    b = find_best_threshold(np.arange(0.05, 1, 0.05), model, dataloader)
    b = 0.4
    best_thres = find_best_threshold_area_and_proba(np.arange(b - 0.05, b + 0.15, 0.05),
                                                    np.arange(0, 5000, 500),
                                                    model, dataloader)

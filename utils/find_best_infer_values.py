from utils.dataset import get_dataloader, get_inference_transforms, read_dataset
from utils.callbacks import DiceCallback, AccuracyCallback

import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from ternausnet.models import UNet16
import cv2 as cv

from joblib import Parallel, delayed

from utils.mobilenetv3 import mobilenetv3


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


def find_best_threshold_binary(range, model, dataloader):
    accuracy = {i: [] for i in range}
    metric = AccuracyCallback(threshold=0)

    for images, targets in tqdm(dataloader, total=len(dataloader)):
        targets = targets.cuda()
        images = images.cuda()
        result =  torch.sigmoid(model(images))
        result = result.detach()
        for threshold in list(accuracy.keys()):
            accuracy[threshold].append(metric.accuracy(result, targets, threshold=threshold).cpu().numpy())

    accuracy = [(key, np.mean(item)) for key, item in accuracy.items()]
    accuracy = sorted(accuracy, reverse=True, key=lambda x: x[1])

    result = accuracy[0][0]
    print(f'Best threshold: {accuracy[0][0]}: {accuracy[0][1]:.5f}')

    return result


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv.threshold(probability, threshold, 1, cv.THRESH_BINARY)[1]
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, borderType=cv.BORDER_CONSTANT, borderValue=0)
    num_component, component = cv.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions


def remove_overlap(result):
    _result = result.copy()

    for i, r0 in enumerate(_result):
        for r1 in _result[i+1:]:
            diff = r0 - r1

            cond = diff != r0
            r0[cond & (diff < 0)] = 0
            r1[cond & (diff > 0)] = 0

    return _result


def func(result, masks, keys, metric=DiceCallback(threshold=0)):
    result_d = {key: [] for key in keys}

    for key in keys:
        proba, area = key
        _result = []

        result = remove_overlap(result)

        for r in result:
            _result.append(post_process(r, proba, area))
        _result = np.stack(_result, axis=0)
        _result = torch.from_numpy(_result)

        result_d[key].append(metric.dice(_result, masks, threshold=proba, activation=None))

    return result_d


def find_best_threshold_area_and_proba(proba_range, area_range, model, dataloader):
    dice = {}
    for p in proba_range:
        for a in area_range:
            dice[(p, a)] = []

    sigmoid = torch.nn.Sigmoid()

    for images, masks in tqdm(dataloader, total=len(dataloader)):
        masks = masks.cuda()
        images = images.cuda()
        result = model(images)
        result = sigmoid(result)
        result = result.detach().cpu().numpy()
        masks = masks.detach().cpu()

        keys = list(dice.keys())

        result = Parallel(n_jobs=len(images))(delayed(func)(r, m, keys) for r, m in zip(result, masks))

        for key in keys:
            for r in result:
                dice[key] += r[key]

    dice = [(key, np.mean(item)) for key, item in dice.items()]
    dice = sorted(dice, reverse=True, key=lambda x: x[1])

    result = dice[0][1]

    for key, val in dice:
        if val < result:
            break
        print(f'Best threshold: {key}: {result:.5f}')

    return result


if __name__ == '__main__':
    # Get dataset
    transforms = get_inference_transforms(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225),)

    df = read_dataset('../dataset/train.csv',
                      '../dataset/train_images',)

    # df = df.sample(10)
    dataloader = get_dataloader(df, transforms,
                                batch_size=8,
                                shuffle=False,
                                num_workers=6,
                                phase='valid',
                                catalyst=False,
                                pin_memory=False,
                                binary=True)

    model = mobilenetv3(1).cuda().eval()
    state = torch.load('/home/druzhinin/HDD/kaggle/kaggle_severstal/logdir/1.2.resnet50_binary_hard_transforms/binary/checkpoints/best.pth')
    model.load_state_dict(state['model_state_dict'])
    del state
    model = model.eval()
    find_best_threshold_binary(np.arange(0.05, 1, 0.05), model, dataloader)

    # ------------------------------------------------------------------------------------------------------------------------------------------

    # df = read_dataset('../dataset/train.csv',
    #                   '../dataset/train_images', )
    # df = df.dropna(subset=[1, 2, 3, 4], how='all')
    # dataloader = get_dataloader(df, transforms,
    #                             batch_size=6,
    #                             shuffle=False,
    #                             num_workers=6,
    #                             phase='valid',
    #                             catalyst=False,
    #                             pin_memory=False,
    #                             binary=True)
    #
    #
    # # Load model
    # model = smp.Unet('resnet50', encoder_weights='imagenet', classes=4, activation=None).cuda().eval()
    # state = torch.load('/home/druzhinin/HDD/kaggle/kaggle_severstal/logdir/1.2.resnet50_binary_hard_transforms/seg/checkpoints/best.pth')
    # # model = UNet16(4, pretrained=True).cuda().eval()
    # # state = torch.load('/home/druzhinin/HDD/kaggle/kaggle_severstal/logdir/1.5.ternausnet/checkpoints/best.pth')
    # model.load_state_dict(state['model_state_dict'])
    # del state
    # model = model.eval()
    #
    # # Find best threshold
    # b = find_best_threshold(np.arange(0.05, 1, 0.05), model, dataloader)
    # b = 0.6
    # best_thres = find_best_threshold_area_and_proba([b],
    #                                                 np.arange(1000, 5001, 500),
    #                                                 model, dataloader)

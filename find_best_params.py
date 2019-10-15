import torch
from utils.dataset import read_dataset, BinaryClassSteelDataset, TrainSteelDataset, MultiClassSteelDataset
import albumentations as A
from torch.utils.data import DataLoader


from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap

import numpy as np

from tqdm import tqdm

from joblib import Parallel, delayed


def get_tta_loaders(df, tta, cls, batch=16):
    datasets = [TtaWrap(cls(df, transforms=t), tfms=t) for t
                in tta]
    loaders = [DataLoader(d, num_workers=6, batch_size=batch, shuffle=False) for d
               in datasets]

    return loaders


def predict_tta_class(models: dict, loaders_batch):
    results = {i: [] for i in models}
    data = None

    for i, data in enumerate(loaders_batch):
        features = data.pop('features').cuda()
        for key, model in models.items():
            results[key].append(torch.sigmoid(model(features)))
        data = data

    # TTA mean
    for key in list(results.keys()):
        preds = torch.stack(results.pop(key))
        preds = torch.mean(preds, dim=0).squeeze()
        results[key] = preds

    return results, data


def find_best_multiclass(df, tta,  model, thresholds):
    loaders = get_tta_loaders(df, tta, MultiClassSteelDataset)

    results = {i: [] for i in thresholds}

    for batch in tqdm(zip(*loaders), total=len(loaders[0])):
        preds, data = predict_tta_class({'model': model}, batch)
        preds = preds['model']
        true = data['targets'].cuda().long()

        for t in thresholds:
            p = (preds > t).long()
            results[t].append((p == true).detach().cpu().numpy().astype(int))

    shape = results[list(results.keys())[0]][0].shape
    accuracy = {cls: {} for cls in range(shape[1])}
    for t, values in results.items():
        values = np.concatenate(values, axis=0)

        for cls in list(accuracy.keys()):
            accuracy[cls][t] = np.sum(values[:, cls]) / len(values[:, cls])

    bests = {}
    for cls, (key, data) in enumerate(accuracy.items()):
        d = [(i, val) for i, val in data.items()]
        d = sorted(d, key=lambda x: x[1])
        print(f'Class: {cls}; Best: {d[-1]} Worse: {d[0]} Mean: {np.mean([i for _, i in d])}')
        bests[cls] = d[-1]

    return bests


def find_best_binary(df, tta,  model, thresholds):
    loaders = get_tta_loaders(df, tta, BinaryClassSteelDataset)

    results = {i: [] for i in thresholds}

    for batch in tqdm(zip(*loaders), total=len(loaders[0])):
        preds, data = predict_tta_class({'model': model}, batch)
        preds = preds['model']
        true = data['targets'].cuda().long()

        for t in thresholds:
            p = (preds > t).long()
            results[t].append((p == true).detach().cpu().numpy().astype(int).flatten())

    accuracy = []
    for t, values in results.items():
        values = np.concatenate(values, axis=0)
        accuracy.append([t, np.sum(values) / len(values)])

    accuracy = sorted(accuracy, key=lambda x: x[1])
    print(f'Best: {accuracy[-1]} Worse: {accuracy[0]} Mean: {np.mean([i for _, i in accuracy])}')

    return accuracy[-1]


def dice(preds, true, eps=1e-7):
    batch_size = len(true)
    outputs = preds.view(batch_size, 4, -1)
    targets = true.view(batch_size, 4, -1)

    intersection = torch.sum(targets * outputs, dim=-1)
    union = torch.sum(targets, dim=-1) + torch.sum(outputs, dim=-1)
    dice = (2 * intersection / (union + eps))

    return dice

def find_best_mask(df, tta,  model, conf_thresholds, min_size_thresholds, batch=16):
    loaders = get_tta_loaders(df, tta, TrainSteelDataset, batch=batch)

    results = {}

    for batch in tqdm(zip(*loaders), total=len(loaders[0])):
        preds, data = predict_tta_class({'model': model}, batch)
        preds = preds['model']
        true = data['targets'].detach().cuda()
        true_not_exists = true.sum(dim=(2, 3)) == 0

        for conf_t in conf_thresholds:
            _preds = (preds > conf_t).float()

            for size_t in min_size_thresholds:
                key = (conf_t, size_t)
                if key not in results:
                    results[key] = []

                data = dice(_preds, true)
                data = torch.where((data < 1e-7) & true_not_exists, torch.ones_like(data), data)
                results[key].append(data)

    for key in list(results.keys()):
        results[key] = torch.cat(results[key], dim=0).mean(dim=0).cpu().numpy()
    _results = {i: [] for i in range(4)}
    for key, item in results.items():
        for i in range(4):
            _results[i].append([key, item[i]])

    for cls, item in _results.items():
        item = sorted(item, key=lambda x: x[1])
        best = item[-1]
        params, d = best
        print(f'Class: {cls}; Best: conf_thres={params[0]:.2f} size_thres={params[1]:.0f} dice={d:.5f}')

    # for name, data in results.items():
    #
    #     d = []
    #     for key, item in data.items():
    #         d.append([key, np.array(item).mean()])
    #
    #     _results[name] = sorted(d, key=lambda x: x[1])
    #
    # print(f'Best catalyst: {_results["catalyst"][-1]}')
    # print(f'Best my: {_results["my"][-1]}')
    #
    # return {'catalyst': _results['catalyst'][-1], 'my': _results['my'][-1]}


if __name__ == '__main__':
    df = read_dataset(
        '/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/dataset/train.csv',
        '/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/dataset/train_images')


    def create_transforms(additional):
        res = list(additional)
        # add necessary transformations
        res.extend([
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ChannelTranspose()
        ])
        res = A.Compose(res)
        return res


    # Different transforms for TTA wrapper
    transforms = [
        [],
        [A.HorizontalFlip(p=1)],
        [A.VerticalFlip(p=1)],
        [A.HorizontalFlip(p=1), A.VerticalFlip(p=1)]
    ]

    transforms = [create_transforms(t) for t in transforms]

    device = 'cuda'


    # print('Multiclass')
    # model = torch.jit.load('/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/download/resnet34-class01/torchscript.pth', map_location=device)
    # find_best_multiclass(df, transforms, model, np.arange(0.05, 1.0, 0.05))
    #
    # print('Binary')
    # model = torch.jit.load('/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/download/12mobilenet-severstal/torchscript.pth', map_location=device)
    # find_best_binary(df, transforms, model, np.arange(0.05, 1.0, 0.05))

    model = torch.jit.load( '/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/download/11resnet50-hard-severstal/torchscript.pth', map_location=device)
    find_best_mask(df, transforms, model, np.arange(0.05, 1.0, 0.05), [500, 1000, 2000, 3000])